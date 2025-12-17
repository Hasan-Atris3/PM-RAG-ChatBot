# app.py
import time
import chainlit as cl

from srd_engine_v2 import SmartKnowledgeBase, ClaudeAnswerer
from db import SessionLocal, User, Chat, Message

claude = ClaudeAnswerer()


@cl.on_chat_start
async def start():
    session = cl.user_session
    db = SessionLocal()
    try:
        # -----------------------
        # USER IDENTIFICATION
        # -----------------------
        if not session.get("user_id"):
            user = User()
            db.add(user)
            db.commit()
            session.set("user_id", user.id)

        user_id = session.get("user_id")

        # -----------------------
        # CHAT SELECTION
        # -----------------------
        chats = db.query(Chat).filter(Chat.user_id == user_id).all()

        actions = [cl.Action(name="new_chat", payload={}, label="➕ New Project Chat")]
        for c in chats[-5:]:
            actions.append(
                cl.Action(
                    name="resume_chat",
                    payload={"chat_id": c.id},
                    label=f"📂 {c.project_name}"
                )
            )

        res = await cl.AskActionMessage(
            content="Choose a chat or start a new one:",
            actions=actions
        ).send()

        if not res:
            return

        # -----------------------
        # NEW CHAT
        # -----------------------
        if res["name"] == "new_chat":
            project_res = await cl.AskUserMessage(
                content="Enter **Project Name**:",
                timeout=300
            ).send()
            if not project_res:
                return
            project_name = project_res["output"]

            learn_res = await cl.AskActionMessage(
                content="Allow this chat to be saved/learned for improving the bot?",
                actions=[
                    cl.Action(name="learn_yes", payload={"v": True}, label="✅ Yes (Enable Learning)"),
                    cl.Action(name="learn_no", payload={"v": False}, label="❌ No (Do Not Learn)"),
                ],
            ).send()
            learning_enabled = bool(learn_res["payload"]["v"]) if learn_res else True

            chat = Chat(user_id=user_id, project_name=project_name, learning_enabled=learning_enabled)
            db.add(chat)
            db.commit()

            session.set("chat_id", chat.id)
            session.set("learning_enabled", chat.learning_enabled)

            engine = SmartKnowledgeBase(chroma_dir="chroma_global_db")
            engine.set_current_project(project_name)
            engine.set_current_chat(chat.id)     # ✅ NEW
            engine.set_current_user(user_id)     # ✅ NEW
            session.set("engine", engine)

            await run_ingestion(engine)

        # -----------------------
        # RESUME CHAT
        # -----------------------
        else:
            chat_id = res["payload"]["chat_id"]
            chat = db.query(Chat).get(chat_id)
            if not chat:
                await cl.Message(content="⚠️ Chat not found.").send()
                return

            session.set("chat_id", chat.id)
            session.set("learning_enabled", chat.learning_enabled)

            engine = SmartKnowledgeBase(chroma_dir="chroma_global_db")
            engine.set_current_project(chat.project_name)
            engine.set_current_chat(chat.id)     # ✅ NEW
            engine.set_current_user(user_id)     # ✅ NEW
            session.set("engine", engine)

            # Restore history
            messages = (
                db.query(Message)
                .filter(Message.chat_id == chat_id)
                .order_by(Message.created_at)
                .all()
            )
            for m in messages:
                await cl.Message(content=m.content, author=m.role).send()

    finally:
        db.close()


async def run_ingestion(engine: SmartKnowledgeBase):
    files = await cl.AskFileMessage(
        content="Upload the **SRD PDF**:",
        accept=["application/pdf"],
        max_size_mb=50,
        timeout=600
    ).send()
    if not files:
        return

    srd_file_path = files[0].path

    res = await cl.AskActionMessage(
        content="Select Diagram Vision Mode:",
        actions=[
            cl.Action(name="qwen", payload={"v": "qwen"}, label="Qwen"),
            cl.Action(name="claude", payload={"v": "claude"}, label="Claude"),
            cl.Action(name="both", payload={"v": "both"}, label="Both"),
            cl.Action(name="none", payload={"v": "none"}, label="None"),
        ]
    ).send()

    mode = res["payload"]["v"] if res else "none"
    use_qwen = mode in ("qwen", "both")
    use_claude = mode in ("claude", "both")

    status = cl.Message(content="🚀 Starting ingestion...")
    await status.send()

    await cl.make_async(engine.process_document_step)(
        srd_file_path, "pdf_text", "SRD Main", False, False
    )

    status.content += "\n✅ SRD indexed"
    await status.update()

    while True:
        add = await cl.AskActionMessage(
            content="Add a diagram?",
            actions=[
                cl.Action(name="yes", payload={}, label="➕ Add"),
                cl.Action(name="done", payload={}, label="Done"),
            ]
        ).send()

        if not add or add["name"] == "done":
            break

        title = await cl.AskUserMessage(content="Diagram title:", timeout=300).send()
        if not title:
            break

        file = await cl.AskFileMessage(
            content="Upload diagram:",
            accept=["image/png", "image/jpeg", "application/pdf"],
            max_size_mb=20,
            timeout=600
        ).send()

        if not file:
            break

        await cl.make_async(engine.process_document_step)(
            file[0].path, "diagram", title["output"], use_qwen, use_claude
        )

        status.content += f"\n🎨 Diagram '{title['output']}' indexed"
        await status.update()

    status.content += "\n🎉 Ingestion complete. Ask questions!"
    await status.update()


@cl.on_message
async def main(message: cl.Message):
    session = cl.user_session
    db = SessionLocal()
    try:
        engine: SmartKnowledgeBase = session.get("engine")
        chat_id = session.get("chat_id")

        if not engine or not chat_id:
            await cl.Message(content="⚠️ Session error. Please refresh.").send()
            return

        # Save user msg
        db.add(Message(chat_id=chat_id, role="user", content=message.content))
        db.commit()

        # Answer
        response = await cl.make_async(engine.generate_smart_response)(message.content, claude)

        # Save assistant msg
        db.add(Message(chat_id=chat_id, role="assistant", content=response))
        db.commit()

        await cl.Message(content=response).send()

        # Feedback
        await cl.Message(
            content="",
            actions=[
                cl.Action(
                    name="correct",
                    payload={"original": message.content},
                    label="🔧 Correct This"
                )
            ]
        ).send()

    finally:
        db.close()


@cl.action_callback("correct")
async def on_correct(action):
    session = cl.user_session
    db = SessionLocal()
    try:
        engine: SmartKnowledgeBase = session.get("engine")
        chat_id = session.get("chat_id")
        learning_enabled = bool(session.get("learning_enabled", True))

        await action.remove()

        res = await cl.AskUserMessage(
            content="Paste the correct information:",
            timeout=600
        ).send()
        if not res:
            return

        # Always store correction text in DB (audit trail)
        db.add(Message(chat_id=chat_id, role="user_feedback", content=res["output"]))
        db.commit()

        # Learn only if allowed
        if learning_enabled:
            engine.learn_from_interaction(action.payload["original"], res["output"])
            await cl.Message(content="✅ Correction saved and learned.").send()
        else:
            await cl.Message(content="✅ Correction saved (learning disabled for this chat).").send()

    finally:
        db.close()
