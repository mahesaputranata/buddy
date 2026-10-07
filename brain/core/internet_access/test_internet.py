from explore_internet import ExploreInternet, ExploreInternetConfig
from internet_providers import (
    ProviderConfig,
    OpenMeteoWeatherProvider,
    RSSNewsProvider,
    DuckDuckGoHTMLSearchProvider,
)

def main():
    pcfg = ProviderConfig(timeout_s=8)

    explorer = ExploreInternet(
        config=ExploreInternetConfig(
            enabled=True,
            default_location=__import__("brain.settings", fromlist=["setting"]).setting("default_location", "Berlin"),
            default_locale="de-DE",
        ),
        weather_provider=OpenMeteoWeatherProvider(pcfg),
        news_provider=RSSNewsProvider(cfg=pcfg),
        search_provider=DuckDuckGoHTMLSearchProvider(pcfg),
    )

    print("=== WEATHER ===")
    print(explorer.get_weather_brief())

    print("\n=== NEWS ===")
    print(explorer.get_news_brief(limit=3))

    print("\n=== SEARCH ===")
    results = explorer.search_web(query="python speech recognition best practices", limit=3)
    print(ExploreInternet.format_search_results(results))

if __name__ == "__main__":
    main()