# 🧪 E2B SANDBOX TEST GUIDE

## ⚠️ IMPORTANT: Test E2B Before Using

Before running your application, **TEST E2B FIRST** to ensure it works.

---

## 📋 Prerequisites

### 1. Get E2B API Key
1. Go to: https://e2b.dev/dashboard
2. Sign up or log in
3. Create a new API key
4. Copy the key (starts with `e2b_`)

### 2. Add to .env.local
Create or update `.env.local`:

```env
E2B_API_KEY=e2b_your_actual_key_here
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
```

### 3. Install Dependencies
```bash
npm install
```

---

## 🧪 Test 1: Simple E2B Test

Run the test script:

```bash
node test-e2b.js
```

### Expected Output:
```
🧪 Testing E2B Sandbox...

✅ E2B_API_KEY found
Key preview: e2b_abc123...

⚡ Creating E2B sandbox...
✅ Sandbox created successfully!
Sandbox ID: sandbox-abc123

📂 Testing file writing...
✅ File written successfully

⚡ Testing command execution...
✅ Command executed successfully
Output: Hello from E2B!

🐍 Testing Python...
✅ Python available
Version: Python 3.x.x

🌐 Testing port forwarding...
✅ Port forwarding works
Host: sandbox-abc123.e2b.dev
URL: https://sandbox-abc123.e2b.dev

🎉 All tests passed!

✅ E2B is working correctly!
You can now use it in your application.
```

### If Test Fails:

#### Error: "E2B_API_KEY not found"
**Solution**: Add E2B_API_KEY to `.env.local`

#### Error: "403 Forbidden"
**Solutions**:
1. Check your API key is valid
2. Get a new key from https://e2b.dev/dashboard
3. Make sure you have credits in your E2B account
4. Verify you're not using a template parameter

#### Error: "Could not resolve host"
**Solution**: Check your internet connection

---

## 🧪 Test 2: E2B Manager Test

Create a test file `test-manager.js`:

```javascript
require('dotenv').config({ path: '.env.local' });

async function testManager() {
  // Import dynamically to handle TypeScript
  const { E2BManager } = await import('./src/lib/e2b.ts');
  
  console.log('🧪 Testing E2BManager...\n');
  
  const manager = new E2BManager();
  
  // Test sandbox creation
  console.log('⚡ Creating sandbox...');
  await manager.createSandbox();
  console.log('✅ Sandbox created:', manager.getSandboxId());
  
  // Test file writing
  console.log('\n📂 Writing files...');
  await manager.writeFiles({
    'test.txt': 'Hello World',
    'test.py': 'print("Hello from Python")'
  });
  console.log('✅ Files written');
  
  // Test command
  console.log('\n⚡ Running command...');
  const result = await manager.runCommand('cat /home/user/test.txt');
  console.log('✅ Command output:', result.stdout);
  
  console.log('\n🎉 E2BManager works correctly!');
}

testManager().catch(console.error);
```

Run:
```bash
node test-manager.js
```

---

## 🧪 Test 3: Full Application Test

### 1. Start Dev Server
```bash
npm run dev
```

### 2. Open Browser
```
http://localhost:3000/ai-workspace
```

### 3. Test Prompt
Enter in chat:
```
Create a simple sentiment analysis model
```

### 4. Watch Console
Open browser DevTools (F12) and check Console tab.

**Expected logs**:
```
✅ E2B Sandbox created: sandbox-abc123
✅ File written: requirements.txt
✅ File written: train.py
✅ File written: app.py
📦 Installing dependencies...
✅ Dependencies installed successfully
🏋️ Starting training...
✅ Training completed successfully
🚀 Deploying FastAPI server...
✅ API deployed at: https://sandbox-abc123.e2b.dev
```

### 5. Check UI

**Chat Area (Left)**:
- Should show ONLY status messages
- NO code blocks
- Progress indicators

**Code Tab (Right)**:
- Should show all generated files
- Tabs for each file
- Syntax highlighting
- Copy buttons

**Sandbox Tab (Right)**:
- Should show live E2B sandbox
- Public URL accessible
- API endpoints working

---

## 🚨 Common Errors & Solutions

### Error: "exit status 1"

**Possible Causes**:
1. E2B_API_KEY not set
2. Invalid API key
3. No credits in E2B account
4. Network issue
5. Sandbox creation failed

**Solutions**:
1. Run `node test-e2b.js` to diagnose
2. Check `.env.local` has E2B_API_KEY
3. Verify API key at https://e2b.dev/dashboard
4. Check browser console for detailed error
5. Check terminal/server logs

### Error: "No Sandbox Active"

**Cause**: Sandbox creation failed silently

**Solution**:
1. Check E2B_API_KEY is valid
2. Run test script first
3. Check browser console for errors
4. Verify E2B account has credits

### Error: "Failed to create E2B sandbox"

**Cause**: E2B API returned error

**Solution**:
1. Check API key is correct
2. Verify internet connection
3. Check E2B service status
4. Try creating sandbox manually at https://e2b.dev/dashboard

---

## ✅ Verification Checklist

Before using the application:

- [ ] E2B_API_KEY added to `.env.local`
- [ ] `node test-e2b.js` passes all tests
- [ ] Sandbox creates successfully
- [ ] Files write successfully
- [ ] Commands execute successfully
- [ ] Port forwarding works
- [ ] Browser console shows no errors
- [ ] Code appears in Code tab only
- [ ] Sandbox preview shows in Sandbox tab

---

## 📊 Test Results

### ✅ Working:
- E2B sandbox creation
- File writing
- Command execution
- Python available
- Port forwarding
- Public URL generation

### ❌ Not Working:
- (List any issues here)

---

## 🆘 Still Having Issues?

### 1. Check E2B Dashboard
https://e2b.dev/dashboard
- Verify API key is active
- Check usage/credits
- View sandbox logs

### 2. Check Browser Console
F12 → Console tab
- Look for red errors
- Check network tab for failed requests

### 3. Check Server Logs
Terminal where `npm run dev` is running
- Look for error messages
- Check for E2B-related errors

### 4. Test Manually
Try creating a sandbox manually:
```javascript
const { Sandbox } = require('@e2b/code-interpreter');
const sandbox = await Sandbox.create();
console.log(sandbox.sandboxId);
```

---

## 📚 Resources

- E2B Documentation: https://e2b.dev/docs
- E2B Dashboard: https://e2b.dev/dashboard
- E2B Examples: https://github.com/e2b-dev/e2b
- Support: https://e2b.dev/discord

---

## 🎯 Next Steps

Once all tests pass:

1. ✅ E2B is working
2. ✅ Apply database schema in Supabase
3. ✅ Test full application
4. ✅ Deploy to production

**Test E2B first, then everything else will work!** 🚀
