import requests
import time
import sys

def keep_service_alive(url, interval_minutes=10):
    """
    Pings a service URL at regular intervals to keep it alive.
    
    Args:
        url: The URL to ping (should include http:// or https://)
        interval_minutes: How often to ping the service (in minutes)
    """
    interval_seconds = interval_minutes * 60
    print(f"Starting keep-alive pings to {url} every {interval_minutes} minutes...")
    
    ping_count = 0
    
    try:
        while True:
            try:
                response = requests.get(url)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                
                if response.status_code == 200:
                    ping_count += 1
                    print(f"[{timestamp}] Ping #{ping_count} successful! Service is alive.")
                else:
                    print(f"[{timestamp}] Warning: Received status code {response.status_code}")
                    
            except requests.RequestException as e:
                print(f"Error pinging service: {e}")
                
            # Sleep for the specified interval
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        print("\nKeep-alive process terminated by user.")
        sys.exit(0)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Keep a web service alive by sending regular ping requests")
    parser.add_argument("url", help="The URL to ping (including the health endpoint)")
    parser.add_argument("--interval", type=int, default=10, help="Ping interval in minutes (default: 10)")
    
    args = parser.parse_args()
    
    # Make sure the URL has the proper format
    if not args.url.startswith(("http://", "https://")):
        args.url = "https://" + args.url
        
    if not args.url.endswith("/health"):
        args.url = args.url.rstrip("/") + "/health"
    
    keep_service_alive(args.url, args.interval) 