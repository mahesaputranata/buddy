from __future__ import annotations

import ipaddress
import json
import os
import platform
import re
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from pathlib import Path
from typing import Callable, Iterable, Optional


DEFAULT_ROVER_HOSTNAME = "ubuntu.local"
DEFAULT_CACHE_PATH = Path(__file__).with_name("rover_connection_cache.json")
DEFAULT_MAC_PREFIXES = (
    "B8:27:EB",  # Raspberry Pi Foundation
    "DC:A6:32",
    "E4:5F:01",
    "D8:3A:DD",
    "2C:CF:67",
    "28:CD:C1",
)


class RoverConnectionDiscovery:
    """
    Finds the rover host before the WebSocket client tries to connect.

    Priority:
      1. Last-known-good cached IP.
      2. Configured mDNS hostname.
      3. Fast local /24 scan filtered by known MAC prefixes.
    """

    def __init__(
        self,
        hostname: str = DEFAULT_ROVER_HOSTNAME,
        cache_path: Path | str = DEFAULT_CACHE_PATH,
        mac_prefixes: Iterable[str] = DEFAULT_MAC_PREFIXES,
        port_probe_timeout_s: float = 0.6,
        scan_timeout_s: float = 3.0,
        logger: Optional[Callable[[str], None]] = print,
    ):
        self.hostname = hostname or DEFAULT_ROVER_HOSTNAME
        self.cache_path = Path(cache_path)
        self.mac_prefixes = tuple(_normalize_mac_prefix(p) for p in mac_prefixes if p)
        self.port_probe_timeout_s = port_probe_timeout_s
        self.scan_timeout_s = scan_timeout_s
        self.logger = logger

    @classmethod
    def from_env(cls, hostname: str | None = None) -> "RoverConnectionDiscovery":
        configured_prefixes = os.getenv("ROVER_MAC_PREFIXES", "")
        prefixes = (
            [p.strip() for p in configured_prefixes.split(",") if p.strip()]
            if configured_prefixes
            else DEFAULT_MAC_PREFIXES
        )
        cache_path = os.getenv("ROVER_DISCOVERY_CACHE", str(DEFAULT_CACHE_PATH))
        return cls(
            hostname=hostname or os.getenv("ROVER_HOSTNAME", DEFAULT_ROVER_HOSTNAME),
            cache_path=cache_path,
            mac_prefixes=prefixes,
        )

    def discover(self, port: int, skip_cache: bool = False) -> Optional[str]:
        if not skip_cache:
            cached_ip = self.load_cached_ip()
            if cached_ip and self._port_open(cached_ip, port):
                self._log(f"[ROVER] Using cached rover IP {cached_ip}")
                return self.cache_success(cached_ip)

        mdns_ip = self.resolve_hostname()
        if mdns_ip and self._port_open(mdns_ip, port):
            self._log(f"[ROVER] Resolved {self.hostname} to {mdns_ip}")
            return self.cache_success(mdns_ip)

        scanned_ip = self.scan_local_subnet(port=port)
        if scanned_ip:
            self._log(f"[ROVER] Found rover by MAC prefix at {scanned_ip}")
            return self.cache_success(scanned_ip)

        return None

    def load_cached_ip(self) -> Optional[str]:
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        ip = data.get("last_known_ip")
        return ip if _is_ip_address(ip) else None

    def cache_success(self, ip: str) -> str:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_known_ip": ip,
            "hostname": self.hostname,
            "updated_at": time.time(),
        }
        self.cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return ip

    def resolve_hostname(self) -> Optional[str]:
        try:
            _, _, addresses = socket.gethostbyname_ex(self.hostname)
        except OSError:
            return None

        for address in addresses:
            if _is_ip_address(address):
                return address
        return None

    def scan_local_subnet(self, port: int) -> Optional[str]:
        if not self.mac_prefixes:
            return None

        local_ip = _get_local_ip()
        if not local_ip:
            return None

        network = ipaddress.ip_network(f"{local_ip}/24", strict=False)
        candidates = [str(ip) for ip in network.hosts() if str(ip) != local_ip]
        deadline = time.monotonic() + self.scan_timeout_s

        with ThreadPoolExecutor(max_workers=64) as pool:
            future_to_ip = {
                pool.submit(_ping_once, ip, max(0.05, min(0.25, deadline - time.monotonic()))): ip
                for ip in candidates
                if time.monotonic() < deadline
            }
            try:
                for future in as_completed(future_to_ip, timeout=self.scan_timeout_s):
                    if time.monotonic() >= deadline:
                        break
                    if not future.result():
                        continue

                    if not future.result():
                        continue
                    ip = future_to_ip[future]
                    if self._port_open(ip, port):         
                        mac = _arp_lookup(ip)
                        if not self.mac_prefixes or (mac and _mac_matches(mac, self.mac_prefixes)):
                            return ip
                        
            except FuturesTimeoutError:
                return None

        return None

    def _port_open(self, host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=self.port_probe_timeout_s):
                return True
        except OSError:
            return False

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)


def _normalize_mac_prefix(prefix: str) -> str:
    return prefix.strip().upper().replace("-", ":")


def _is_ip_address(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _get_local_ip() -> Optional[str]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return None


def _ping_once(ip: str, timeout_s: float) -> bool:
    timeout_ms = max(100, int(timeout_s * 1000))
    if platform.system().lower().startswith("win"):
        cmd = ["ping", "-n", "1", "-w", str(timeout_ms), ip]
    else:
        cmd = ["ping", "-c", "1", "-W", str(max(1, int(timeout_s))), ip]

    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout_s + 0.2)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _arp_lookup(ip: str) -> Optional[str]:
    try:
        result = subprocess.run(["arp", "-a", ip], capture_output=True, text=True, timeout=0.5)
    except (OSError, subprocess.TimeoutExpired):
        return None

    match = re.search(r"([0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}", result.stdout)
    return match.group(0).upper().replace("-", ":") if match else None


def _mac_matches(mac: str, prefixes: Iterable[str]) -> bool:
    normalized = mac.upper().replace("-", ":")
    return any(normalized.startswith(prefix) for prefix in prefixes)
