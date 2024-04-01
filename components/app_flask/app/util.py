import re

def extract_info_from_endpoint(url):
    """Extracts location, endpoint, and project information from a given AIPlatform URL.

    Args:
        url: The AIPlatform URL string.

    Returns:
        A dictionary containing the extracted values:
            locations: The region.
            endpoints: The endpoint ID.
            projects: The project ID.
    """

    pattern = r"\/projects\/([^\/]+)\/locations\/([^\/]+)\/endpoints\/([^\/]+)\/operations\/([^\/]+)"
    print(pattern)
    match = re.search(pattern, url)
    print(match)
    if match:
        return {
            "projects": match.group(1),
            "locations": match.group(2),
            "endpoints": match.group(3)
        }
    else:
        return None  # Or you could raise an exception if the URL is invalid