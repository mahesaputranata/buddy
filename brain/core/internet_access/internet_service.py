from brain.core.internet_access.explore_internet import ExploreInternet, ExploreInternetConfig
from brain.core.internet_access.internet_providers import ProviderConfig, OpenMeteoWeatherProvider, RSSNewsProvider, DuckDuckGoHTMLSearchProvider

def get_explorer(enabled: bool = True) -> ExploreInternet:
    pcfg = ProviderConfig(timeout_s=8)

    return ExploreInternet(
        config=ExploreInternetConfig(
            enabled=enabled,
            default_location=__import__("brain.settings", fromlist=["setting"]).setting("default_location", "Berlin"),
            default_locale="de-DE",
        ),
        weather_provider=OpenMeteoWeatherProvider(pcfg),
        news_provider=RSSNewsProvider(cfg=pcfg),
        search_provider=DuckDuckGoHTMLSearchProvider(pcfg),
    )


