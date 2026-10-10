import paramiko
import os
import shlex
import time

class RoverSSHClient:
    def __init__(self, host, username="ubuntu", key_path="~/.ssh/id_ed25519",
                 ros_distro="humble",
                 script_path = "rover_ws/src/rover_bridge/rover_bridge/rover_server.py",
                 log_path = "/tmp/rover_server.log",
                 workspace = "rover_ws"):

        self.host = host
        self.username = username
        self.key_path = os.path.expanduser(key_path)
        self.ros_distro = ros_distro
        self.script_path = script_path
        self.log_path = log_path
        self.workspace = workspace

    # server activation
    def _connect(self):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(self.host,
                       username=self.username,
                       key_filename=self.key_path,
                       timeout=8,)

        return client

    def _run(self, command: str):
        client = self._connect()
        try:
            stdin, stdout, stderr = client.exec_command(command)
            exit_code = stdout.channel.recv_exit_status()
            out = stdout.read().decode(errors="replace").strip()
            err = stderr.read().decode(errors="replace").strip()
            return exit_code, out, err
        finally:
            client.close()

    def check_rover_server_status(self) -> bool:
        command = "pgrep -af '[r]over_server.py'"
        exit_code, out, err = self._run(command)
        return bool(out)

    def get_rover_server_status(self) -> bool:
        command = "pgrep -af '[r]over_server.py'"
        exit_code, out, err = self._run(command)
        return out

    def stop_rover_server(self) -> bool:
        command = "pkill -f '[r]over_server.py'"
        exit_code, out, err = self._run(command)
        return exit_code in (0, 1)

    def start_rover_server(self) -> bool:
        script = shlex.quote(self.script_path)
        log = shlex.quote(self.log_path)

        command = f"nohup python3 {script} > {log} 2>&1 < /dev/null &"
        self._run(command)

        time.sleep(1)
        return self.check_rover_server_status()

    def restart_rover_server(self) -> bool:
        self.stop_rover_server()
        time.sleep(1)
        return self.start_rover_server()

    def tail_rover_server_log(self, lines=40) -> str: # later need a good summary logic for buddy to understand it
        log = shlex.quote(self.log_path)
        command = f"tail -n {int(lines)} {log}"
        exit_code, out, err = self._run(command)
        return out or err

    # ros2 management
    def _ros_command(self, inner_command: str) -> str:
        setup = (
            f"source /opt/ros/{self.ros_distro}/setup.bash &&"
            f"source {self.workspace}/install/setup.bash &&"
        )
        return "bash -lc " + shlex.quote(setup + inner_command)

    def _start_background(self, process_name: str, ros_command: str, log_path: str):
        cmd = self._ros_command(
            f"nohup {ros_command} > {shlex.quote(log_path)} 2>&1 < /dev/null &"
        )
        self._run(cmd)
        time.sleep(1)
        return self.is_process_running(process_name)

    def is_process_running(self, pattern: str) -> bool:
        cmd = f"pgrep -af {shlex.quote(pattern)}"
        exit_code, out, err = self._run(cmd)
        return bool(out)

    def get_process(self, pattern:str) -> str:
        cmd = f"pgrep -af {shlex.quote(pattern)}"
        exit_code, out, err = self._run(cmd)
        return out

    def stop_process(self, pattern:str) -> str:
        cmd = f"pkill -f {shlex.quote(pattern)}"
        exit_code, out, err = self._run(cmd)
        return exit_code in (0,1)


    # ros2 activation for bridge node
    def start_bridge_node(self) -> bool:
        return self._start_background(
            process_name="ros2 run rover_bridge bridge_node",
            ros_command="ros2 run rover_bridge bridge_node",
            log_path="/tmp/rover_bridge_node.log",
        )

    def stop_bridge_node(self) -> bool:
        return self.stop_process("ros2 run rover_bridge bridge_node")

    def restart_bridge_node(self) -> bool:
        self.stop_bridge_node()
        time.sleep(1)
        return self.start_bridge_node()

    def bridge_node_status(self) -> bool:
        return self.is_process_running("ros2 run rover_bridge bridge_node")

    # ros2 activation for control node
    def start_control_node(self) -> bool:
        return self._start_background(
            process_name="ros2 launch rover_control control.launch.py",
            ros_command="ros2 launch rover_control control.launch.py",
            log_path="/tmp/rover_control.log",
        )

    def stop_control_node(self) -> bool:
        return self.stop_process("ros2 launch rover_control control.launch.py")

    def restart_control_node(self) -> bool:
        self.stop_control_node()
        time.sleep(1)
        return self.start_control_node()

    def control_node_status(self) -> bool:
        return self.is_process_running("ros2 launch rover_control control.launch.py")

    # Both

    def start_ros2_stack(self) -> dict:
        control_ok = self.start_control_node()
        bridge_ok = self.start_bridge_node()
        return {
            "control": control_ok,
            "bridge": bridge_ok,
        }

    def stop_ros2_stack(self) -> dict:
        bridge_stopped = self.stop_bridge_node()
        control_stopped = self.stop_control_node()
        return {
            "bridge": bridge_stopped,
            "control": control_stopped,
        }

    def restart_ros2_stack(self) -> dict:
        self.stop_ros2_stack()
        time.sleep(1)
        return self.start_ros2_stack()

    def ros2_stack_status(self) -> dict:
        return {
            "control": self.control_node_status(),
            "bridge": self.bridge_node_status(),
        }

    # log
    def tail_log(self, log_path: str, lines: int = 50) -> str:
        cmd = f"tail -n {int(lines)} {shlex.quote(log_path)}"
        exit_code, out, err = self._run(cmd)
        return out or err

    def bridge_log(self, lines: int = 50) -> str:
        return self.tail_log("/tmp/rover_bridge_node.log", lines)

    def control_log(self, lines: int = 50) -> str:
        return self.tail_log("/tmp/rover_control.log", lines)

    # record and unrecord video

    def start_recording(self, session_id: str = None) -> str:
        session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        cmd = (
            f"nohup python3 ~/rover_recorder.py {shlex.quote(session_id)} "
            f"> /tmp/rover_recorder_{session_id}.log 2>&1 < /dev/null & "
            f"echo $!"  # capture PID so stop_recording can target it precisely
        )
        exit_code, out, err = self._run(cmd)
        self._active_recording_pid = out.strip()
        self._active_session_id = session_id
        return session_id

    def stop_recording(self) -> bool:
        if not getattr(self, "_active_recording_pid", None):
            return False
        cmd = f"kill -TERM {shlex.quote(self._active_recording_pid)}"
        exit_code, out, err = self._run(cmd)
        time.sleep(1)  # let SIGTERM handler flush/close files cleanly
        return exit_code in (0, 1)

    def fetch_recording(self, session_id: str, local_dir: str) -> dict:
        remote_video = f"recordings/{session_id}.h264"
        remote_log = f"recordings/{session_id}_detections.jsonl"

        client = self._connect()
        try:
            sftp = client.open_sftp()
            local_video = f"{local_dir}/{session_id}.h264"
            local_log = f"{local_dir}/{session_id}_detections.jsonl"
            sftp.get(remote_video, local_video)
            sftp.get(remote_log, local_log)
            sftp.close()
            return {"video": local_video, "log": local_log}
        finally:
            client.close()

    def check_disk_space(self) -> dict:
        cmd = "df -h ~ | tail -1 && du -sh ~/recordings 2>/dev/null || echo '0'"
        exit_code, out, err = self._run(cmd)
        # parse `out` — df line + du line — return structured dict
        lines = out.strip().splitlines()
        df_parts = lines[0].split()
        return {
            "total": df_parts[1],
            "used": df_parts[2],
            "free": df_parts[3],
            "free_pct": 100 - int(df_parts[4].rstrip('%')),
            "recordings_size": lines[1] if len(lines) > 1 else "unknown",
        }

    def list_recordings(self) -> list[str]:
        cmd = "ls -1t ~/recordings/*.h264 2>/dev/null"
        exit_code, out, err = self._run(cmd)
        return out.strip().splitlines() if out else []

    def delete_recordings(self, session_ids: list[str]) -> bool:
        # DESTRUCTIVE — must go through the confirmation gate before ever being called
        files = " ".join(
            shlex.quote(f"~/recordings/{sid}.h264") for sid in session_ids
        ) + " " + " ".join(
            shlex.quote(f"~/recordings/{sid}_detections.jsonl") for sid in session_ids
        )
        cmd = f"rm -f {files}"
        exit_code, out, err = self._run(cmd)
        return exit_code == 0