import asyncio

async def wait_for_mock_awaited(
    mock,
    num_awaited: int,
    timeout: int = 5,
):
    async def _poll_mock(mock):
        while not mock.await_count == num_awaited:
            await asyncio.sleep(0.1)
    await asyncio.wait_for(_poll_mock(mock), timeout=timeout)