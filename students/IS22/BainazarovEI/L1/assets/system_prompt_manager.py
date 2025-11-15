import os
from database import SystemPrompt

def save_system_prompt(db_manager, name, content):
    session = db_manager.get_session()
    try:
        session.query(SystemPrompt).update({SystemPrompt.is_active: False})
        new_prompt = SystemPrompt(name=name, content=content, is_active=True)
        session.add(new_prompt)
        session.commit()
        print(f"Промпт '{name}' сохранен и активирован!")
    except Exception as e:
        session.rollback()
        print(f"Ошибка при сохранении промпта: {e}")
    finally:
        session.close()

def get_active_system_prompt(db_manager):
    env_prompt = os.getenv('DEEPSEEK_SYSTEM_PROMPT')
    if env_prompt:
        return env_prompt
    
    session = db_manager.get_session()
    try:
        active_prompt = session.query(SystemPrompt).filter(SystemPrompt.is_active == True).first()
        return active_prompt.content if active_prompt else "You are a helpful assistant"
    finally:
        session.close()

def manage_system_prompts(db_manager):
    while True:
        print("\n=== Управление системными промптами ===")
        print("1. Просмотреть сохраненные промпты")
        print("2. Добавить новый промпт")
        print("3. Активировать промпт")
        print("4. Вернуться в главное меню")
        
        choice = input("Выберите действие: ")
        
        match choice:
            case "1":
                view_prompts(db_manager)
            case "2":
                add_prompt(db_manager)
            case "3":
                activate_prompt(db_manager)
            case "4":
                break
            case _:
                print("Неверный выбор. Попробуйте снова.")
         
def view_prompts(db_manager):
    session = db_manager.get_session()
    try:
        prompts = session.query(SystemPrompt).order_by(SystemPrompt.id).all()
        
        if not prompts:
            print("Нет сохраненных промптов.")
            return
        
        print("\nСохраненные промпты:")
        for prompt in prompts:
            status = "АКТИВЕН" if prompt.is_active else "неактивен"
            print(f"{prompt.id}. {prompt.name} [{status}]")
            print()
    finally:
        session.close()

def add_prompt(db_manager):
    print("\nДобавление нового системного промпта:")
    name = input("Введите название промпта: ")
    
    if not name.strip():
        print("Название не может быть пустым!")
        return
    
    print("Введите содержание промпта (введите 'END' на новой строке для завершения):")
    
    content_lines = []
    while True:
        line = input()
        if line.strip() == 'END':
            break
        content_lines.append(line)
    
    content = '\n'.join(content_lines)
    
    if not content.strip():
        print("Содержание промпта не может быть пустым!")
        return
    
    save_system_prompt(db_manager, name, content)

def activate_prompt(db_manager):
    view_prompts(db_manager)
    
    session = db_manager.get_session()
    try:
        prompts = session.query(SystemPrompt).all()
        
        if not prompts:
            print("Нет доступных промптов для активации.")
            return
        
        try:
            prompt_id = int(input("Введите ID промпта для активации: "))
            
            session.query(SystemPrompt).update({SystemPrompt.is_active: False})
            
            target_prompt = session.query(SystemPrompt).filter(SystemPrompt.id == prompt_id).first()
            
            if target_prompt:
                target_prompt.is_active = True
                session.commit()
                print(f"Промпт '{target_prompt.name}' активирован!")
            else:
                print("Промпт с таким ID не найден.")
                session.rollback()
                
        except ValueError:
            print("Неверный формат ID.")
    finally:
        session.close()