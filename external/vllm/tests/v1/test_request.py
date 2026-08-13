# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus


def create_request(arrival_time: float) -> Request:
    return Request(
        request_id="request",
        prompt_token_ids=[],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=arrival_time,
    )


def test_request_activity_initialized_from_arrival_time():
    request = create_request(arrival_time=100.0)

    assert request.last_activity_time == request.arrival_time


def test_request_mark_activity_with_explicit_time():
    request = create_request(arrival_time=100.0)

    request.mark_activity(125.0)

    assert request.last_activity_time == 125.0


def test_request_idle_time():
    request = create_request(arrival_time=100.0)
    request.mark_activity(125.0)

    assert request.get_idle_time(current_time=130.0) == 5.0


def test_request_idle_time_never_negative():
    request = create_request(arrival_time=100.0)
    request.mark_activity(125.0)

    assert request.get_idle_time(current_time=120.0) == 0.0


def test_request_status_fmt_str():
    """Test that the string representation of RequestStatus is correct."""
    assert f"{RequestStatus.WAITING}" == "WAITING"
    assert (
        f"{RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR}"
        == "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR"
    )
    assert f"{RequestStatus.WAITING_FOR_REMOTE_KVS}" == "WAITING_FOR_REMOTE_KVS"
    assert f"{RequestStatus.WAITING_FOR_STREAMING_REQ}" == "WAITING_FOR_STREAMING_REQ"
    assert f"{RequestStatus.RUNNING}" == "RUNNING"
    assert f"{RequestStatus.PREEMPTED}" == "PREEMPTED"
    assert f"{RequestStatus.FINISHED_STOPPED}" == "FINISHED_STOPPED"
    assert f"{RequestStatus.FINISHED_LENGTH_CAPPED}" == "FINISHED_LENGTH_CAPPED"
    assert f"{RequestStatus.FINISHED_ABORTED}" == "FINISHED_ABORTED"
    assert f"{RequestStatus.FINISHED_IGNORED}" == "FINISHED_IGNORED"
