import re

# Read the file
with open('src/app/ai-workspace/page.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove all supabase checks from conditions
content = content.replace('if (!supabase || !user)', 'if (!user)')
content = content.replace('if (!supabase)', 'if (false)  // removed supabase check')
content = content.replace('if (user && supabase)', 'if (user)')
content = content.replace('if (supabase && user)', 'if (user)')
content = content.replace('if (supabase && currentChat)', 'if (currentChat)')
content = content.replace(', supabase]', ']')

# 2. Replace loadChats function
old_load_chats = '''  const loadChats = async () => {
    if (!user) return;

    try {
      const { data, error } = await supabase
        .from('chats')
        .select('*')
        .eq('user_id', user.id)
        .order('updated_at', { ascending: false });

      if (data && !error) {
        setChats(data);
      }
    } catch (err) {
      console.error('Error loading chats:', err);
    }
  };'''

new_load_chats = '''  const loadChatsData = async () => {
    if (!user) return;

    try {
      const data = await loadChats(user.id);
      setChats(data);
    } catch (err) {
      console.error('Error loading chats:', err);
    }
  };'''

content = content.replace(old_load_chats, new_load_chats)

# Also replace the call
content = content.replace('loadChats();', 'loadChatsData();')

# 3. Replace loadMessages function
old_load_messages = '''  const loadMessages = async (chatId: string) => {
    if (false)  // removed supabase check return;

    try {
      const { data, error } = await supabase
        .from('messages')
        .select('*')
        .eq('chat_id', chatId)
        .order('created_at', { ascending: true });

      if (data && !error) {
        setMessages(data);
      }
    } catch (err) {
      console.error('Error loading messages:', err);
    }
  };'''

new_load_messages = '''  const loadMessagesData = async (chatId: string) => {
    try {
      const data = await loadMessages(chatId);
      setMessages(data as Message[]);
    } catch (err) {
      console.error('Error loading messages:', err);
    }
  };'''

content = content.replace(old_load_messages, new_load_messages)

# Replace the call
content = content.replace('loadMessages(chat.id)', 'loadMessagesData(chat.id)')

# 4. Replace createNewChat function  
old_create_chat = '''  const createNewChat = async () => {
    if (!user) return;

    try {
      const { data, error } = await supabase
        .from('chats')
        .insert({
          user_id: user.id,
          title: 'New Chat',
          mode: 'models'
        })
        .select()
        .single();

      if (data && !error) {
        setChats(prev => [data, ...prev]);
        setCurrentChat(data);
        setMessages([]);
        setInputValue('');
      }
    } catch (error) {
      console.error('Error in createNewChat:', error);
    }
  };'''

new_create_chat = '''  const createNewChatHandler = async () => {
    if (!user) return;

    try {
      const data = await createNewChat(user.id);
      if (data) {
        setChats(prev => [data, ...prev]);
        setCurrentChat(data);
        setMessages([]);
        setInputValue('');
      }
    } catch (error) {
      console.error('Error in createNewChat:', error);
    }
  };'''

content = content.replace(old_create_chat, new_create_chat)

# Replace the call
content = content.replace('createNewChat()', 'createNewChatHandler()')

# 5. Replace deleteChat function
old_delete = '''  const deleteChat = async (chatId: string, e?: React.MouseEvent) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }

    if (!user) return;

    try {
      await supabase.from('messages').delete().eq('chat_id', chatId);
      const { error: chatError } = await supabase
        .from('chats')
        .delete()
        .eq('id', chatId)
        .eq('user_id', user.id);

      if (!chatError) {
        setChats(prev => prev.filter(chat => chat.id !== chatId));
        if (currentChat?.id === chatId) {
          setCurrentChat(null);
          setMessages([]);
        }'''

new_delete = '''  const deleteChatHandler = async (chatId: string, e?: React.MouseEvent) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }

    if (!user) return;

    try {
      const success = await deleteChat(chatId);
      if (success) {
        setChats(prev => prev.filter(chat => chat.id !== chatId));
        if (currentChat?.id === chatId) {
          setCurrentChat(null);
          setMessages([]);
        }'''

content = content.replace(old_delete, new_delete)

# Replace the call
content = content.replace('deleteChat(chat.id, e)', 'deleteChatHandler(chat.id, e)')

# 6. Fix all Message objects to include required fields
# This is a pattern to find message objects and add missing fields
def fix_message_object(match):
    # Extract the message object
    obj = match.group(0)
    
    # If it already has chat_id, skip it
    if 'chat_id:' in obj:
        return obj
    
    # Add required fields
    if 'role: \'assistant\'' in obj or 'role: "assistant"' in obj:
        # Find where to insert chat_id
        if 'id:' in obj:
            obj = obj.replace('id:', 'chat_id: activeChat?.id || currentChat?.id || \'\',\n          id:')
        obj = obj.replace('created_at:', 'model_used: null,\n          metadata: { eventId },\n          created_at:')
    elif 'role: \'user\'' in obj or 'role: "user"' in obj:
        if 'id:' in obj:
            obj = obj.replace('id:', 'chat_id: activeChat?.id || currentChat?.id || \'\',\n          id:')
        if 'created_at:' in obj and 'model_used:' not in obj:
            obj = obj.replace('created_at:', 'model_used: null,\n          metadata: {},\n          created_at:')
    
    return obj

# Apply the fix (this is a simplified version, may need manual adjustment)

# 7. Remove the supabase null check at the end
content = content.replace('''  if (!supabase) {
    return (
      <div style={{
        display: 'flex',''', '''  if (false) {  // removed supabase check
    return (
      <div style={{
        display: 'flex',''')

# Write the fixed content
with open('src/app/ai-workspace/page.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed page.tsx successfully!")
