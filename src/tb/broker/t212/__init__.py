"""The Trading 212 adapter."""

from tb.broker.t212.client import AuthScheme, ClientConfig, T212Client
from tb.broker.t212.endpoints import SPECS, Endpoint, EndpointSpec, spec_for
from tb.broker.t212.errors import (
    AuthError,
    BrokerError,
    BrokerHttpError,
    RateLimited,
    SchemaDriftError,
    TransportError,
    UnknownOrderState,
)
from tb.broker.t212.ratelimit import NullGovernor, RateGovernor, RateLimitHeaders, RateLimitTimeout
from tb.broker.t212.raw_archive import RawArchive

__all__ = [
    "SPECS",
    "AuthError",
    "AuthScheme",
    "BrokerError",
    "BrokerHttpError",
    "ClientConfig",
    "Endpoint",
    "EndpointSpec",
    "NullGovernor",
    "RateGovernor",
    "RateLimitHeaders",
    "RateLimitTimeout",
    "RateLimited",
    "RawArchive",
    "SchemaDriftError",
    "T212Client",
    "TransportError",
    "UnknownOrderState",
    "spec_for",
]
