# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from starlette.routing import Match

from vllm.entrypoints.serve.instrumentator.metrics import _vllm_safe_get_route_name


class _RouteWithoutPath:
    def matches(self, scope):
        return Match.FULL, {}


class _RouteWithPath:
    path = "/v1/models"

    def matches(self, scope):
        return Match.FULL, {}


def test_safe_get_route_name_ignores_wrapper_routes_without_path():
    route_name = _vllm_safe_get_route_name(
        {"type": "http", "path": "/v1/models"},
        [_RouteWithoutPath()],
    )

    assert route_name is None


def test_safe_get_route_name_preserves_normal_route_paths():
    route_name = _vllm_safe_get_route_name(
        {"type": "http", "path": "/v1/models"},
        [_RouteWithPath()],
    )

    assert route_name == "/v1/models"
