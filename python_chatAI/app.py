import logging
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from cozepy import COZE_CN_BASE_URL, ChatEventType, Coze, Message, TokenAuth, MessageType

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Coze WebSocket 聊天 API (带追问建议)")

# 加载配置
try:
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
        COZE_API_TOKEN = config.get("coze_api_token", "")
        COZE_BOT_ID = config.get("bot_id", "")
    logger.info("Config loaded successfully")
except FileNotFoundError:
    logger.error("Missing config.json file")
    raise ValueError("Missing config.json file")

# 初始化 Coze 客户端
coze = Coze(
    auth=TokenAuth(token=COZE_API_TOKEN),
    base_url=COZE_CN_BASE_URL,
)
logger.info("Coze client initialized")

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket 连接已建立")

    try:
        while True:
            # 接收前端消息
            data = await websocket.receive_text()
            # logger.debug(f"收到客户端原始数据: {data}")

            try:
                data = json.loads(data)
                user_id = data.get("user_id")
                user_input = data.get("user_input", "")
                if not user_id:
                    logger.warning("缺少 user_id")
                    await websocket.send_json({"error": "user_id is required"})
                    continue
                # logger.info(f"处理用户输入: user_id={user_id}, input='{user_input}'")
            except json.JSONDecodeError:
                logger.error("无效的 JSON 格式")
                await websocket.send_json({"error": "Invalid JSON format"})
                continue

            # 调用 Coze 流式 API
            # logger.info(f"调用 Coze API，bot_id={COZE_BOT_ID}, 开启追问建议")
            stream = coze.chat.stream(
                bot_id=COZE_BOT_ID,
                user_id=user_id,
                additional_messages=[Message.build_user_question_text(user_input)],
                parameters={"suggest_followup": True}
            )

            # 处理流式响应
            for event in stream:
                logger.debug(f"事件详情: event={event.event}, message_type={getattr(event.message, 'type', None)}")

                if event.event == ChatEventType.CONVERSATION_MESSAGE_DELTA:
                    if event.message and event.message.type == MessageType.ANSWER:
                        # logger.debug(f"流式内容: {event.message.content}")
                        await websocket.send_json({"type": "content", "data": event.message.content})

                elif event.event == ChatEventType.CONVERSATION_MESSAGE_COMPLETED:
                    if event.message and event.message.type == MessageType.FOLLOW_UP:
                        # logger.info(f"收到追问建议: {event.message.content}")
                        await websocket.send_json({"type": "followup", "data": event.message.content})

                elif event.event == ChatEventType.CONVERSATION_CHAT_COMPLETED:
                    logger.info("对话完成")
                    await websocket.send_json({"type": "done", "data": "[DONE]"})

    except WebSocketDisconnect:
        logger.info("客户端断开连接")
    except Exception as e:
        logger.error(f"发生错误: {str(e)}", exc_info=True)
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()
        logger.info("WebSocket 连接已关闭")