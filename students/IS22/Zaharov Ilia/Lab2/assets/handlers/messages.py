import logging
from aiogram import F
from utils.loader import dp
from utils.database import db
from aiogram.types import Message
from aiogram.enums import ContentType
from utils.mistral import get_response, client


@dp.message(F.content_type == ContentType.TEXT)
async def message_handler(message: Message) -> None:
    try:
        user_id = message.from_user.id
        user_name = message.from_user.first_name or message.from_user.username
        
        await db.add_user(
            user_id=user_id,
            username=message.from_user.username,
            first_name=message.from_user.first_name,
            last_name=message.from_user.last_name
        )
        
        await db.add_message(user_id, "user", message.text)

        response = await get_response(message.text, user_id, user_name, client, use_context=True)
        
        await db.add_message(user_id, "assistant", response)
        
        await message.answer(response)
    except Exception as e:
        logging.error(f"Error occured: {e}")
        await message.answer("Произошла ошибка при получении ответа")


@dp.message(F.content_type == ContentType.PHOTO)
async def handle_photos(message: Message) -> None:
    """Обработчик для изображений"""
    try:
        user_name = message.from_user.first_name or message.from_user.username
        
        await db.add_user(
            user_id=message.from_user.id,
            username=message.from_user.username,
            first_name=message.from_user.first_name,
            last_name=message.from_user.last_name
        )
        
        await db.add_message(
            message.from_user.id, 
            "user", 
            f"[Изображение] {message.caption if message.caption else ''}"
        )
        
        response = f"{user_name}, вы отправили картинку!"
        if message.caption:
            response += f"\n\nС подписью: \"{message.caption}\""
        
        await db.add_message(message.from_user.id, "assistant", response)
        await message.answer(response)
        
    except Exception as e:
        logging.error(f"Error occured while processing photo: {e}")
        await message.answer("Произошла ошибка при обработке изображения")


@dp.message(F.content_type.in_({
    ContentType.DOCUMENT, 
    ContentType.VIDEO, 
    ContentType.AUDIO, 
    ContentType.VOICE
}))
async def handle_other_media(message: Message) -> None:
    """Обработчик для других типов медиа-файлов"""
    try:
        user_name = message.from_user.first_name or message.from_user.username
        media_type = ""
        
        if message.content_type == ContentType.DOCUMENT:
            media_type = "документ"
        elif message.content_type == ContentType.VIDEO:
            media_type = "видео"
        elif message.content_type == ContentType.AUDIO:
            media_type = "аудио"
        elif message.content_type == ContentType.VOICE:
            media_type = "голосовое сообщение"
        
        await db.add_user(
            user_id=message.from_user.id,
            username=message.from_user.username,
            first_name=message.from_user.first_name,
            last_name=message.from_user.last_name
        )
        
        await db.add_message(
            message.from_user.id, 
            "user", 
            f"[{media_type.capitalize()}] {message.caption if message.caption else ''}"
        )
        
        response = f"{user_name}, вы отправили {media_type}!"
        if message.caption:
            response += f"\n\nС подписью: \"{message.caption}\""
        
        await db.add_message(message.from_user.id, "assistant", response)
        await message.answer(response)
        
    except Exception as e:
        logging.error(f"Error occured while processing {message.content_type}: {e}")
        await message.answer("Произошла ошибка при обработке медиа-файла")