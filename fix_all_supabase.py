import re

# Read the file
with open('src/app/ai-workspace/page.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all supabase.from('ai_models').insert() calls
content = re.sub(
    r'await supabase\.from\([\'"]ai_models[\'"]\)\.insert\(([^)]+)\)',
    r'await saveAIModel(\1)',
    content
)

# Replace all supabase.from('messages').insert() calls
content = re.sub(
    r'await supabase\.from\([\'"]messages[\'"]\)\.insert\(\{([^}]+)\}\)',
    r'await insertMessage({\1})',
    content
)

# Replace all supabase.from('chats').insert() calls with createNewChat
# This one is more complex, need to handle it carefully
def replace_chat_insert(match):
    return '''const newChatData = createChatObject(user.id, content.slice(0, 50) + (content.length > 50 ? '...' : ''), 'models');
        const data = await createNewChat(user.id);
        if (!data) throw new Error('Failed to create chat');
        chatToUse = data;'''

content = re.sub(
    r'const \{ data, error \} = await supabase\s*\.from\([\'"]chats[\'"]\)\s*\.insert\(\{[^}]+\}\)\s*\.select\(\)\s*\.single\(\);',
    replace_chat_insert,
    content,
    flags=re.DOTALL
)

# Replace supabase.from('chats').update() calls
content = re.sub(
    r'await supabase\s*\.from\([\'"]chats[\'"]\)\s*\.update\(\{([^}]+)\}\)\s*\.eq\([\'"]id[\'"]\s*,\s*([^)]+)\)',
    r'await updateChat(\2, {\1})',
    content
)

# Fix the sendMessage function - replace the chat creation part
old_chat_create = '''      try {
        const newChatData = createChatObject(user.id, content.slice(0, 50) + (content.length > 50 ? '...' : ''), 'models');
        const data = await createNewChat(user.id);
        if (!data) throw new Error('Failed to create chat');
        chatToUse = data;
          .from('chats')
          .insert({
            user_id: user.id,
            title: content.slice(0, 50) + (content.length > 50 ? '...' : ''),
            mode: 'models'
          })
          .select()
          .single();

        if (error || !data) {
          throw new Error('Failed to create new chat');
        }

        chatToUse = data;'''

new_chat_create = '''      try {
        const data = await createNewChat(user.id);
        if (!data) {
          throw new Error('Failed to create new chat');
        }
        chatToUse = data;'''

content = content.replace(old_chat_create, new_chat_create)

# Write the fixed content
with open('src/app/ai-workspace/page.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed all supabase calls!")
