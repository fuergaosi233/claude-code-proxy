from src.conversion.request_converter import convert_claude_user_message
from src.models.claude import ClaudeMessage


def test_convert_mixed_text_image_and_video_content() -> None:
    message = ClaudeMessage.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the media."},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "image-data",
                    },
                },
                {
                    "type": "video",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/video.mp4",
                        "detail": "high",
                        "fps": 2,
                        "max_long_side_pixel": 1920,
                    },
                },
            ],
        }
    )

    assert convert_claude_user_message(message) == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the media."},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,image-data"},
            },
            {
                "type": "video_url",
                "video_url": {
                    "url": "https://example.com/video.mp4",
                    "detail": "high",
                    "fps": 2,
                    "max_long_side_pixel": 1920,
                },
            },
        ],
    }


def test_convert_base64_video_content() -> None:
    message = ClaudeMessage.model_validate(
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "source": {
                        "type": "base64",
                        "media_type": "video/mp4",
                        "data": "video-data",
                    },
                }
            ],
        }
    )

    assert convert_claude_user_message(message) == {
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {"url": "data:video/mp4;base64,video-data"},
            }
        ],
    }
