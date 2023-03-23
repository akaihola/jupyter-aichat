import pytest


def filter_request_headers(request):
    """Filter out sensitive headers from the request before recording it."""
    request.headers = {
        header: value
        for header, value in request.headers.items()
        if header.lower() == "content-type"
    }
    return request


def filter_response_headers(response):
    """Filter out sensitive headers from the response before recording it."""
    from pprint import pprint

    pprint(response["headers"])
    response["headers"] = {
        header: value
        for header, value in response["headers"].items()
        if header.lower() in ["content-type", "openai-model"]
    }
    return response


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "before_record_request": filter_request_headers,
        "before_record_response": filter_response_headers,
    }
