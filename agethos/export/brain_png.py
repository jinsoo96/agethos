""".brain.png — 스테가노그래피 뇌 패키징.

PNG 이미지에 .brain ZIP 데이터를 임베딩.
공유/배포 시 이미지로 보이지만 실제로는 뇌 데이터를 포함.

원리:
    PNG 파일은 IEND 청크 이후 데이터를 무시함.
    → PNG 이미지 뒤에 .brain ZIP 데이터를 append.
    → 이미지 뷰어는 정상 PNG로 인식, agethos는 뒤의 ZIP을 추출.

구조:
    [PNG 이미지 데이터] [AGETHOS_MARKER (8 bytes)] [.brain ZIP 데이터]
"""

from __future__ import annotations

import struct
from pathlib import Path

# 매직 바이트: agethos .brain 데이터 시작 마커
AGETHOS_MARKER = b"AGETHOS\x00"

# PNG IEND 청크 끝 시그니처
PNG_IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"


async def pack_brain_png(
    brain,
    image_path: str | Path,
    output_path: str | Path,
    include_history: bool = True,
) -> Path:
    """.brain 데이터를 PNG 이미지에 임베딩.

    Args:
        brain: Brain 인스턴스.
        image_path: 베이스 PNG 이미지 경로.
        output_path: 출력 .brain.png 파일 경로.
        include_history: 대화 기록 포함 여부.

    Returns:
        저장된 파일의 Path.

    Raises:
        ValueError: 유효한 PNG 파일이 아닌 경우.
    """
    from agethos.export.brain_file import pack_brain

    image_path = Path(image_path)
    output_path = Path(output_path)

    # PNG 유효성 검사
    image_data = image_path.read_bytes()
    if not image_data[:8] == b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a valid PNG file: {image_path}")

    # .brain ZIP을 메모리에 생성
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".brain", delete=True) as tmp:
        tmp_path = Path(tmp.name)

    await pack_brain(brain, tmp_path, include_history=include_history)
    brain_data = tmp_path.read_bytes()
    tmp_path.unlink(missing_ok=True)

    # 크기 정보 (8 bytes, big-endian unsigned long long)
    size_bytes = struct.pack(">Q", len(brain_data))

    # 합성: PNG + MARKER + SIZE + BRAIN_ZIP
    output_path.write_bytes(image_data + AGETHOS_MARKER + size_bytes + brain_data)
    return output_path


async def unpack_brain_png(
    png_path: str | Path,
    llm,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs,
):
    """.brain.png에서 Brain 복원.

    Args:
        png_path: .brain.png 파일 경로.
        llm: LLM 프로바이더 문자열 또는 어댑터 인스턴스.

    Returns:
        복원된 Brain 인스턴스.

    Raises:
        ValueError: agethos 데이터가 없는 경우.
    """
    from agethos.export.brain_file import unpack_brain

    png_path = Path(png_path)
    data = png_path.read_bytes()

    # 마커 위치 탐색
    marker_pos = data.find(AGETHOS_MARKER)
    if marker_pos == -1:
        raise ValueError(f"No agethos brain data found in: {png_path}")

    # 크기 읽기
    size_start = marker_pos + len(AGETHOS_MARKER)
    brain_size = struct.unpack(">Q", data[size_start:size_start + 8])[0]

    # ZIP 데이터 추출
    brain_start = size_start + 8
    brain_data = data[brain_start:brain_start + brain_size]

    # 임시 파일로 저장 후 unpack
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".brain", delete=False) as tmp:
        tmp.write(brain_data)
        tmp_path = Path(tmp.name)

    try:
        brain = await unpack_brain(
            tmp_path, llm,
            model=model, api_key=api_key, base_url=base_url,
            **kwargs,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return brain


def extract_image(png_path: str | Path, output_path: str | Path) -> Path:
    """`.brain.png`에서 순수 PNG 이미지만 추출.

    Args:
        png_path: .brain.png 파일 경로.
        output_path: 출력 PNG 파일 경로.

    Returns:
        저장된 이미지 Path.
    """
    png_path = Path(png_path)
    output_path = Path(output_path)
    data = png_path.read_bytes()

    marker_pos = data.find(AGETHOS_MARKER)
    if marker_pos == -1:
        # 마커 없으면 전체가 이미지
        output_path.write_bytes(data)
    else:
        output_path.write_bytes(data[:marker_pos])

    return output_path


def has_brain_data(png_path: str | Path) -> bool:
    """PNG 파일에 agethos 뇌 데이터가 포함되어 있는지 확인."""
    data = Path(png_path).read_bytes()
    return AGETHOS_MARKER in data
