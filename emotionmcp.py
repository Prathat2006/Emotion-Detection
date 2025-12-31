import asyncio
import base64
import io
from typing import Any
import numpy as np
import cv2
from PIL import Image
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

# Import your emotion detection function
from detect import detect_emotion

# Create MCP server instance
server = Server("emotion-detection-server")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool is exposed as a callable operation for clients.
    """
    return [
        types.Tool(
            name="detect-emotion",
            description="Detect facial emotion from an image. Accepts image file path only. For best results, provide the full absolute path to the image file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Full file path to the image file (e.g., /home/user/image.jpg or C:\\Users\\user\\image.jpg)"
                    }
                },
                "required": ["image_path"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    """
    if name != "detect-emotion":
        raise ValueError(f"Unknown tool: {name}")

    if not arguments:
        raise ValueError("Missing arguments")

    image_path = arguments.get("image_path")

    if not image_path:
        raise ValueError("image_path is required")

    try:
        # Load image from file path
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image from path: {image_path}\nPlease ensure:\n1. The file path is correct and absolute\n2. The file exists and is readable\n3. The file is a valid image format (jpg, png, etc.)")

        # Run emotion detection
        result = detect_emotion(image)

        # Format response
        if "error" in result and result["error"]:
            response_text = f"❌ {result['error']}"
        else:
            response_text = f"""✅ Emotion Detection Results:
            
📊 Faces Detected: {result.get('faces_detected', 1)}
😊 Emotion: {result['emotion']}
🎯 Confidence: {result['confidence']}%
"""

        return [
            types.TextContent(
                type="text",
                text=response_text
            )
        ]

    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        return [
            types.TextContent(
                type="text",
                text=error_msg
            )
        ]


async def main():
    """Run the MCP server using stdio transport."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="emotion-detection-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())