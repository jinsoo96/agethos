""".brain ZIP, .brain.png, Transplant, 임베딩 추상화 테스트 — v0.8.0."""

import json
import struct
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agethos.brain import Brain
from agethos.embedding import resolve_embedder
from agethos.embedding.base import EmbeddingAdapter
from agethos.export.brain_file import (
    _generate_fingerprint_svg,
    extract_fingerprint,
    inspect_brain,
    pack_brain,
    unpack_brain,
)
from agethos.export.brain_png import (
    AGETHOS_MARKER,
    extract_image,
    has_brain_data,
    pack_brain_png,
    unpack_brain_png,
)
from agethos.export.transplant import (
    AutoGenTransplant,
    CrewAITransplant,
    LangGraphTransplant,
    TransplantAdapter,
    transplant,
)
from agethos.models import (
    MemoryNode,
    MentalModel,
    NodeType,
    OceanTraits,
    PersonaLayer,
    PersonaSpec,
    SocialPattern,
)


# ── Helpers ──


def _make_persona(**kwargs) -> PersonaSpec:
    defaults = dict(
        name="TestBrain",
        ocean=OceanTraits(openness=0.8, conscientiousness=0.7, extraversion=0.3,
                          agreeableness=0.6, neuroticism=0.4),
        l0_innate=PersonaLayer(traits={"age": "30", "occupation": "Engineer"}),
        tone="Concise and analytical",
        values=["Code quality", "Efficiency"],
        behavioral_rules=["Think before responding"],
    )
    defaults.update(kwargs)
    return PersonaSpec(**defaults)


def _make_mock_llm():
    llm = MagicMock()
    llm.generate_with_history = AsyncMock(return_value="Mock response")
    llm.generate = AsyncMock(return_value="Mock response")
    return llm


def _make_brain(**kwargs) -> Brain:
    persona = kwargs.pop("persona", _make_persona())
    llm = kwargs.pop("llm", _make_mock_llm())
    return Brain(persona=persona, llm=llm, **kwargs)


def _make_minimal_png() -> bytes:
    """1x1 투명 PNG 생성."""
    import struct
    import zlib

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = zlib.compress(b"\x00\x00\x00\x00")
    idat = _chunk(b"IDAT", raw)
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


# ══════════════════════════════════════════════════════════════
#  1. Neural Fingerprint SVG
# ══════════════════════════════════════════════════════════════


class TestNeuralFingerprint:
    def test_generates_valid_svg(self):
        persona = _make_persona()
        persona.init_emotion_from_ocean()
        memories = [
            MemoryNode(description=f"memory {i}", importance=5.0)
            for i in range(10)
        ]
        svg = _generate_fingerprint_svg(persona, memories)
        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "TestBrain" in svg

    def test_svg_with_no_memories(self):
        persona = _make_persona()
        svg = _generate_fingerprint_svg(persona, [])
        assert "Memories: 0" in svg

    def test_svg_with_no_emotion(self):
        persona = _make_persona()
        persona.emotion = None
        svg = _generate_fingerprint_svg(persona, [])
        assert "</svg>" in svg

    def test_svg_ocean_values_present(self):
        persona = _make_persona()
        svg = _generate_fingerprint_svg(persona, [])
        assert "0.80" in svg  # openness
        assert "0.70" in svg  # conscientiousness

    def test_svg_many_memories(self):
        persona = _make_persona()
        memories = [MemoryNode(description=f"m{i}") for i in range(200)]
        svg = _generate_fingerprint_svg(persona, memories)
        assert "Memories: 200" in svg


# ══════════════════════════════════════════════════════════════
#  2. .brain ZIP 포맷
# ══════════════════════════════════════════════════════════════


class TestBrainZip:
    async def test_pack_and_unpack(self, tmp_path):
        brain = _make_brain()
        # 기억 추가
        await brain._memory.store.save(MemoryNode(description="test memory 1", importance=7.0))
        await brain._memory.store.save(MemoryNode(description="test memory 2", importance=3.0))
        # 패턴 추가
        brain._social_patterns.append(SocialPattern(
            context="meeting", effective_strategy="listen first", confidence=0.8,
        ))
        # 멘탈 모델 추가
        brain._mental_models["Alice"] = MentalModel(
            target="Alice", believed_goals=["learn Python"],
        )
        # 대화 기록
        brain._history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

        path = tmp_path / "test.brain"
        await pack_brain(brain, path)

        # 파일 검증
        assert path.exists()
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            assert "persona.json" in names
            assert "memories.jsonl" in names
            assert "patterns.json" in names
            assert "mental_models.json" in names
            assert "history.json" in names
            assert "fingerprint.svg" in names

            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["name"] == "TestBrain"
            assert manifest["stats"]["memory_count"] == 2
            assert manifest["stats"]["pattern_count"] == 1
            assert manifest["stats"]["mental_model_count"] == 1

        # Unpack
        restored = await unpack_brain(path, _make_mock_llm())
        assert restored.persona.name == "TestBrain"
        assert restored.persona.ocean.openness == 0.8
        all_mems = await restored.memory.store.get_all()
        assert len(all_mems) == 2
        assert len(restored.social_patterns) == 1
        assert "Alice" in restored.mental_models
        assert len(restored.history) == 2

    async def test_pack_without_fingerprint(self, tmp_path):
        brain = _make_brain()
        path = tmp_path / "nofp.brain"
        await pack_brain(brain, path, include_fingerprint=False)
        with zipfile.ZipFile(path) as zf:
            assert "fingerprint.svg" not in zf.namelist()

    async def test_pack_without_history(self, tmp_path):
        brain = _make_brain()
        brain._history = [{"role": "user", "content": "hi"}]
        path = tmp_path / "nohist.brain"
        await pack_brain(brain, path, include_history=False)
        with zipfile.ZipFile(path) as zf:
            history = json.loads(zf.read("history.json"))
            assert history == []

    async def test_inspect_brain(self, tmp_path):
        brain = _make_brain()
        path = tmp_path / "inspect.brain"
        await pack_brain(brain, path)
        info = inspect_brain(path)
        assert info["name"] == "TestBrain"
        assert "manifest.json" in info["files"]
        assert info["total_size_bytes"] > 0
        assert info["compressed_size_bytes"] > 0

    async def test_extract_fingerprint(self, tmp_path):
        brain = _make_brain()
        path = tmp_path / "fp.brain"
        await pack_brain(brain, path)
        svg = extract_fingerprint(path)
        assert svg is not None
        assert "<svg" in svg

    async def test_extract_fingerprint_missing(self, tmp_path):
        brain = _make_brain()
        path = tmp_path / "nofp.brain"
        await pack_brain(brain, path, include_fingerprint=False)
        assert extract_fingerprint(path) is None

    async def test_brain_pack_method(self, tmp_path):
        """Brain.pack() / Brain.unpack() 통합 테스트."""
        brain = _make_brain()
        path = tmp_path / "method.brain"
        result = await brain.pack(str(path))
        assert Path(result).exists()

        restored = await Brain.unpack(str(path), _make_mock_llm())
        assert restored.persona.name == "TestBrain"

    async def test_empty_brain_pack(self, tmp_path):
        """기억/패턴 없는 빈 Brain도 패킹 가능."""
        brain = _make_brain()
        path = tmp_path / "empty.brain"
        await pack_brain(brain, path)
        restored = await unpack_brain(path, _make_mock_llm())
        assert restored.persona.name == "TestBrain"
        mems = await restored.memory.store.get_all()
        assert len(mems) == 0


# ══════════════════════════════════════════════════════════════
#  3. .brain.png 스테가노그래피
# ══════════════════════════════════════════════════════════════


class TestBrainPng:
    async def test_pack_and_unpack_png(self, tmp_path):
        brain = _make_brain()
        await brain._memory.store.save(MemoryNode(description="png test"))

        # PNG 이미지 생성
        png_path = tmp_path / "base.png"
        png_path.write_bytes(_make_minimal_png())

        # Pack
        output_path = tmp_path / "brain.brain.png"
        await pack_brain_png(brain, png_path, output_path)
        assert output_path.exists()

        # 파일이 여전히 유효한 PNG
        data = output_path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

        # has_brain_data 확인
        assert has_brain_data(output_path)
        assert not has_brain_data(png_path)

        # Unpack
        restored = await unpack_brain_png(output_path, _make_mock_llm())
        assert restored.persona.name == "TestBrain"
        mems = await restored.memory.store.get_all()
        assert len(mems) == 1

    async def test_extract_image(self, tmp_path):
        brain = _make_brain()
        png_path = tmp_path / "base.png"
        png_data = _make_minimal_png()
        png_path.write_bytes(png_data)

        output_path = tmp_path / "brain.brain.png"
        await pack_brain_png(brain, png_path, output_path)

        # 이미지만 추출
        extracted = tmp_path / "extracted.png"
        extract_image(output_path, extracted)
        assert extracted.read_bytes() == png_data

    async def test_invalid_png_raises(self, tmp_path):
        brain = _make_brain()
        not_png = tmp_path / "fake.png"
        not_png.write_bytes(b"not a png file")
        with pytest.raises(ValueError, match="Not a valid PNG"):
            await pack_brain_png(brain, not_png, tmp_path / "out.brain.png")

    async def test_no_brain_data_raises(self, tmp_path):
        plain_png = tmp_path / "plain.png"
        plain_png.write_bytes(_make_minimal_png())
        with pytest.raises(ValueError, match="No agethos brain data"):
            await unpack_brain_png(plain_png, _make_mock_llm())

    async def test_brain_pack_png_method(self, tmp_path):
        """Brain.pack_png() / Brain.unpack_png() 통합 테스트."""
        brain = _make_brain()
        png_path = tmp_path / "base.png"
        png_path.write_bytes(_make_minimal_png())
        output_path = tmp_path / "method.brain.png"

        result = await brain.pack_png(str(png_path), str(output_path))
        assert Path(result).exists()

        restored = await Brain.unpack_png(str(output_path), _make_mock_llm())
        assert restored.persona.name == "TestBrain"

    def test_extract_image_no_marker(self, tmp_path):
        """마커 없는 일반 PNG에서 이미지 추출."""
        png_data = _make_minimal_png()
        plain = tmp_path / "plain.png"
        plain.write_bytes(png_data)
        out = tmp_path / "out.png"
        extract_image(plain, out)
        assert out.read_bytes() == png_data


# ══════════════════════════════════════════════════════════════
#  4. Transplant Adapter
# ══════════════════════════════════════════════════════════════


class TestTransplantAdapter:
    def test_base_adapter(self):
        brain = _make_brain()
        adapter = TransplantAdapter(brain)
        assert adapter.brain is brain
        prompt = adapter._render_system_prompt()
        assert "TestBrain" in prompt

    def test_crewai_transplant_creates_config(self):
        brain = _make_brain()
        t = CrewAITransplant(brain)
        prompt = t._render_system_prompt()
        assert "TestBrain" in prompt
        assert "Code quality" in prompt

    async def test_crewai_sync_after_task(self):
        brain = _make_brain()
        t = CrewAITransplant(brain)
        # observe를 모킹
        brain.observe = AsyncMock()
        await t.sync_after_task("Task completed successfully")
        brain.observe.assert_called_once()

    def test_autogen_transplant(self):
        brain = _make_brain()
        t = AutoGenTransplant(brain)
        prompt = t._render_system_prompt()
        assert "TestBrain" in prompt

    def test_langgraph_transplant(self):
        brain = _make_brain()
        t = LangGraphTransplant(brain)
        node_func = t.as_node()
        assert callable(node_func)

    async def test_langgraph_node_func(self):
        brain = _make_brain()
        t = LangGraphTransplant(brain)
        node_func = t.as_node()

        # tuple 메시지
        state = {"messages": [("user", "Hello!")]}
        result = await node_func(state)
        assert "messages" in result
        assert result["messages"][0][0] == "assistant"

    async def test_langgraph_node_with_dict_messages(self):
        brain = _make_brain()
        t = LangGraphTransplant(brain)
        node_func = t.as_node()

        state = {"messages": [{"role": "user", "content": "Hi!"}]}
        result = await node_func(state)
        assert result["messages"][0][0] == "assistant"

    def test_langgraph_state_snapshot(self):
        brain = _make_brain()
        t = LangGraphTransplant(brain)
        snap = t.get_state_snapshot()
        assert snap["brain_name"] == "TestBrain"
        assert "brain_emotion" in snap
        assert "brain_system_prompt" in snap

    def test_transplant_convenience_unknown_framework(self):
        brain = _make_brain()
        with pytest.raises(ValueError, match="Unknown framework"):
            transplant(brain, "unknown_framework")

    def test_brain_transplant_method_unknown(self):
        brain = _make_brain()
        with pytest.raises(ValueError, match="Unknown framework"):
            brain.transplant("tensorflow")


# ══════════════════════════════════════════════════════════════
#  5. 임베딩 추상화
# ══════════════════════════════════════════════════════════════


class TestEmbeddingAbstraction:
    def test_resolve_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            resolve_embedder("unknown_provider")

    def test_resolve_openai_import_error(self):
        """OpenAI 미설치 시 ImportError."""
        # openai가 설치되어 있으면 건너뛰기
        try:
            import openai
            pytest.skip("openai is installed")
        except ImportError:
            with pytest.raises(ImportError):
                resolve_embedder("openai")

    def test_embedding_adapter_abstract(self):
        """EmbeddingAdapter는 직접 인스턴스화 불가."""
        with pytest.raises(TypeError):
            EmbeddingAdapter()

    def test_brain_build_with_embedder_string(self):
        """embedder를 문자열로 전달 시 resolve 시도."""
        # sentence-transformers 미설치이면 ImportError
        try:
            import sentence_transformers
            brain = Brain.build(
                persona={"name": "Test", "ocean": {"O": 0.5}},
                llm=_make_mock_llm(),
                embedder="sentence-transformer",
            )
            assert brain._memory._embedder is not None
        except ImportError:
            with pytest.raises(ImportError):
                Brain.build(
                    persona={"name": "Test", "ocean": {"O": 0.5}},
                    llm=_make_mock_llm(),
                    embedder="sentence-transformer",
                )

    def test_brain_build_with_embedder_instance(self):
        """embedder를 인스턴스로 전달."""
        class DummyEmbedder(EmbeddingAdapter):
            async def embed(self, texts):
                return [[0.0] * 10 for _ in texts]
            @property
            def dimension(self):
                return 10

        brain = Brain.build(
            persona={"name": "Test", "ocean": {"O": 0.5}},
            llm=_make_mock_llm(),
            embedder=DummyEmbedder(),
        )
        assert brain._memory._embedder is not None

    def test_resolve_embedder_aliases(self):
        """sentence_transformer, sbert 등 별칭도 동작."""
        for alias in ("sentence-transformer", "sentence-transformers", "sbert", "sentence_transformer"):
            try:
                import sentence_transformers
                embedder = resolve_embedder(alias)
                assert embedder is not None
            except ImportError:
                with pytest.raises(ImportError):
                    resolve_embedder(alias)


# ══════════════════════════════════════════════════════════════
#  6. 통합 시나리오
# ══════════════════════════════════════════════════════════════


class TestIntegration:
    async def test_full_lifecycle(self, tmp_path):
        """Brain 생성 → 기억 축적 → .brain 패킹 → 복원 → .brain.png → 복원."""
        # 1. Brain 생성 + 기억
        brain = _make_brain()
        for i in range(5):
            await brain._memory.store.save(
                MemoryNode(description=f"experience {i}", importance=float(i + 1))
            )
        brain._social_patterns.append(
            SocialPattern(context="work", effective_strategy="be polite")
        )
        brain._mental_models["Bob"] = MentalModel(target="Bob", believed_goals=["ship feature"])

        # 2. .brain 패킹
        brain_path = tmp_path / "full.brain"
        await brain.pack(str(brain_path))

        # 3. .brain → 복원
        restored1 = await Brain.unpack(str(brain_path), _make_mock_llm())
        mems1 = await restored1.memory.store.get_all()
        assert len(mems1) == 5
        assert len(restored1.social_patterns) == 1
        assert "Bob" in restored1.mental_models

        # 4. .brain.png 패킹
        png_path = tmp_path / "base.png"
        png_path.write_bytes(_make_minimal_png())
        png_out = tmp_path / "full.brain.png"
        await restored1.pack_png(str(png_path), str(png_out))

        # 5. .brain.png → 복원
        restored2 = await Brain.unpack_png(str(png_out), _make_mock_llm())
        mems2 = await restored2.memory.store.get_all()
        assert len(mems2) == 5
        assert restored2.persona.name == "TestBrain"
        assert restored2.persona.ocean.openness == 0.8

    async def test_inspect_after_pack(self, tmp_path):
        brain = _make_brain()
        await brain._memory.store.save(MemoryNode(description="m1"))
        path = tmp_path / "ins.brain"
        await brain.pack(str(path))
        info = inspect_brain(path)
        assert info["format_version"] == "1.0"
        assert info["stats"]["memory_count"] == 1
