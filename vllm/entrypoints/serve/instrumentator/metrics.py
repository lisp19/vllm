# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import prometheus_client
import regex as re
from fastapi import FastAPI, Response
from prometheus_client import make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator import routing as instrumentator_routing
from starlette.routing import Match, Mount

from vllm.v1.metrics.prometheus import get_prometheus_registry


class PrometheusResponse(Response):
    media_type = prometheus_client.CONTENT_TYPE_LATEST


def _vllm_safe_get_route_name(scope, routes, route_name=None):
    """Compat shim for instrumentator on route trees with wrapper nodes.

    Newer FastAPI route trees may contain wrapper objects such as
    `_IncludedRouter` that implement `matches(...)` but do not expose a
    `.path` attribute. `prometheus_fastapi_instrumentator` still assumes
    `.path` is always present, which turns every request into a 500 before it
    reaches the actual API handler. When that happens, we fall back to letting
    the middleware use the raw request path instead of crashing.
    """

    for route in routes:
        match, child_scope = route.matches(scope)
        route_path = getattr(route, "path", None)
        if match == Match.FULL:
            if route_path is None:
                return route_name
            route_name = route_path
            child_scope = {**scope, **child_scope}
            if isinstance(route, Mount) and route.routes:
                child_route_name = _vllm_safe_get_route_name(
                    child_scope,
                    route.routes,
                    route_name,
                )
                if child_route_name is None:
                    route_name = None
                else:
                    route_name += child_route_name
            return route_name
        if match == Match.PARTIAL and route_name is None and route_path is not None:
            route_name = route_path
    return route_name


def _install_route_name_compat() -> None:
    # Patch the library helper in-process so metrics middleware stays usable
    # across FastAPI route-tree changes.
    instrumentator_routing._get_route_name = _vllm_safe_get_route_name


def attach_router(app: FastAPI):
    """Mount prometheus metrics to a FastAPI app."""

    registry = get_prometheus_registry()
    _install_route_name_compat()

    # `response_class=PrometheusResponse` is needed to return an HTTP response
    # with header "Content-Type: text/plain; version=0.0.4; charset=utf-8"
    # instead of the default "application/json" which is incorrect.
    # See https://github.com/trallnag/prometheus-fastapi-instrumentator/issues/163#issue-1296092364
    Instrumentator(
        excluded_handlers=[
            "/metrics",
            "/health",
            "/load",
            "/ping",
            "/version",
            "/server_info",
        ],
        registry=registry,
    ).add().instrument(app).expose(app, response_class=PrometheusResponse)

    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)
