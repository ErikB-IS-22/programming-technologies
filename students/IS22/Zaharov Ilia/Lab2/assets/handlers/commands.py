import logging
from utils.loader import dp
from utils.database import db
from aiogram.types import Message
from aiogram.filters import CommandStart, Command
from config import SYSTEM_PROMPT, MAX_HISTORY_MESSAGES


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    try:
        await db.add_user(
            user_id=message.from_user.id,
            username=message.from_user.username,
            first_name=message.from_user.first_name,
            last_name=message.from_user.last_name
        )
        
        user_name = message.from_user.first_name or message.from_user.username
        await message.answer(f"Привет, {user_name}! Я твой бот-ассистент! Можешь задавать мне вопросы, и я буду отвечать на них. \n\n"
                            f"Я помню контекст наших предыдущих сообщений (до {MAX_HISTORY_MESSAGES} сообщений). "
                            f"Если хочешь начать разговор заново, используй команду /reset-context.\n\n"
                            f"Также я умею реагировать на изображения и другие медиа-файлы!")
        
    except Exception as e:
        logging.error(f"Error occured: {e}")


@dp.message(Command("prompt"))
async def command_prompt_handler(message: Message) -> None:
    try:
        prompt_preview = SYSTEM_PROMPT[:200] + "..." if len(SYSTEM_PROMPT) > 200 else SYSTEM_PROMPT
        await message.answer(f"📝 Текущий системный промпт:\n\n{prompt_preview}")
    except Exception as e:
        logging.error(f"Error occured: {e}")
        await message.answer("Произошла ошибка при получении промпта")


@dp.message(Command("stats"))
async def command_stats_handler(message: Message) -> None:
    """Команда для просмотра статистики сообщений"""
    try:
        user_id = message.from_user.id
        message_count = await db.get_message_count(user_id)
        user_name = message.from_user.first_name or message.from_user.username
        
        await message.answer(
            f"📊 Статистика для {user_name}:\n"
            f"• Сообщений в истории: {message_count}\n"
            f"• Используется для контекста: {min(message_count, MAX_HISTORY_MESSAGES)} сообщений\n"
            f"• Максимум сообщений в контексте: {MAX_HISTORY_MESSAGES}"
        )
    except Exception as e:
        logging.error(f"Error occured: {e}")
        await message.answer("Произошла ошибка при получении статистики")


@dp.message(Command("history"))
async def command_history_handler(message: Message) -> None:
    """Команда для просмотра последних сообщений"""
    try:
        user_id = message.from_user.id
        messages = await db.get_user_messages(user_id, limit=5)
        
        if not messages:
            await message.answer("История сообщений пуста.")
            return
        
        history_text = "📜 Последние сообщения:\n\n"
        for msg in messages:
            role_emoji = "👤" if msg.role == "user" else "🤖"
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            history_text += f"{role_emoji} {msg.role}: {content_preview}\n\n"
        
        await message.answer(history_text)
    except Exception as e:
        logging.error(f"Error occured: {e}")
        await message.answer("Произошла ошибка при получении истории")


@dp.message(Command("clear"))
async def command_clear_handler(message: Message) -> None:
    """Команда для очистки истории сообщений (алиас для /reset-context)"""
    try:
        user_id = message.from_user.id
        deleted_count = await db.clear_user_messages(user_id)
        await message.answer(f"🗑️ История сообщений очищена. Удалено сообщений: {deleted_count}\n\n"
                           "Теперь я буду отвечать без учета предыдущего контекста.")
    except Exception as e:
        logging.error(f"Error occured: {e}")
        await message.answer("Произошла ошибка при очистке истории")


@dp.message(Command("reset-context"))
async def command_reset_context_handler(message: Message) -> None:
    """Команда для сброса контекста диалога (основная команда по заданию)"""
    try:
        user_id = message.from_user.id
        deleted_count = await db.clear_user_messages(user_id)
        await message.answer(f"🔄 Контекст диалога сброшен. Удалено сообщений: {deleted_count}\n\n"
                           "Теперь я буду отвечать без учета предыдущего контекста.\n"
                           "Мы можем начать разговор заново!")
    except Exception as e:
        logging.error(f"Error occured: {e}")
        await message.answer("Произошла ошибка при сбросе контекста")


@dp.message(Command("context"))
async def command_context_handler(message: Message) -> None:
    """Команда для просмотра текущих настроек контекста"""
    try:
        user_id = message.from_user.id
        message_count = await db.get_message_count(user_id)
        
        await message.answer(
            f"🔍 Настройки контекста:\n"
            f"• Максимум сообщений в контексте: {MAX_HISTORY_MESSAGES}\n"
            f"• Ваших сообщений в базе: {message_count}\n"
            f"• Используется для контекста: {min(message_count, MAX_HISTORY_MESSAGES)} сообщений\n\n"
            f"Используйте /reset-context чтобы очистить историю и начать диалог заново."
        )
    except Exception as e:
        logging.error(f"Error occured: {e}")
        await message.answer("Произошла ошибка при получении информации о контексте")
