"""Tests for types module."""

from eurydice import AudioFormat, Voice


class TestVoice:
    """Tests for Voice enum."""

    def test_all_voices_exist(self):
        """Verify all 8 voices are defined."""
        voices = list(Voice)
        assert len(voices) == 8
        assert Voice.TARA in voices
        assert Voice.LEAH in voices
        assert Voice.JESS in voices
        assert Voice.LEO in voices
        assert Voice.DAN in voices
        assert Voice.MIA in voices
        assert Voice.ZAC in voices
        assert Voice.ZOE in voices

    def test_from_string_valid(self):
        """Test valid voice name conversion."""
        assert Voice.from_string("tara") == Voice.TARA
        assert Voice.from_string("leo") == Voice.LEO
        assert Voice.from_string("LEO") == Voice.LEO  # Case insensitive

    def test_from_string_invalid(self):
        """Test invalid voice name falls back to default."""
        assert Voice.from_string("invalid") == Voice.LEO
        assert Voice.from_string("") == Voice.LEO

    def test_list_all(self):
        """Test listing all voice names."""
        names = Voice.list_all()
        assert len(names) == 8
        assert "leo" in names
        assert "tara" in names


class TestAudioFormat:
    """Tests for AudioFormat enum."""

    def test_formats_exist(self):
        """Verify audio formats are defined."""
        assert AudioFormat.WAV.value == "wav"
        assert AudioFormat.RAW.value == "raw"


class TestAudioResult:
    """Tests for AudioResult dataclass."""

    def test_create_audio_result(self, sample_audio):
        """Test creating an AudioResult."""
        assert sample_audio.audio_data == b"test audio data"
        assert sample_audio.duration == 1.0
        assert sample_audio.format == AudioFormat.WAV
        assert sample_audio.sample_rate == 24000
        assert sample_audio.voice == Voice.LEO
        assert sample_audio.cached is False

    def test_to_base64(self, sample_audio):
        """Test base64 encoding."""
        b64 = sample_audio.to_base64()
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_save_and_load(self, sample_audio, tmp_path):
        """Test saving audio to file."""
        path = str(tmp_path / "test.wav")
        sample_audio.save(path)

        with open(path, "rb") as f:
            data = f.read()

        assert data == sample_audio.audio_data

    def test_as_dict(self, sample_audio):
        """Test dictionary conversion."""
        d = sample_audio.as_dict()
        assert d["duration"] == 1.0
        assert d["format"] == "wav"
        assert d["sample_rate"] == 24000
        assert d["voice"] == "leo"
        assert d["cached"] is False
        assert "audio_data" in d
