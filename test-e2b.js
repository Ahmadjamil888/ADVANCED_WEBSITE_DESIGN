// Test E2B Sandbox Creation
// Run with: node test-e2b.js

require('dotenv').config({ path: '.env.local' });
const { Sandbox } = require('@e2b/code-interpreter');

async function testE2B() {
  console.log('🧪 Testing E2B Sandbox...\n');
  
  // Check API key
  if (!process.env.E2B_API_KEY) {
    console.error('❌ E2B_API_KEY not found in .env.local');
    console.log('Please add: E2B_API_KEY=your_key_here');
    process.exit(1);
  }
  
  console.log('✅ E2B_API_KEY found');
  console.log('Key preview:', process.env.E2B_API_KEY.substring(0, 10) + '...\n');
  
  try {
    console.log('⚡ Creating E2B sandbox...');
    const sandbox = await Sandbox.create();
    
    console.log('✅ Sandbox created successfully!');
    console.log('Sandbox ID:', sandbox.sandboxId);
    
    // Test file writing
    console.log('\n📂 Testing file writing...');
    await sandbox.files.write('/home/user/test.txt', 'Hello from E2B!');
    console.log('✅ File written successfully');
    
    // Test command execution
    console.log('\n⚡ Testing command execution...');
    const result = await sandbox.commands.run('cat /home/user/test.txt');
    console.log('✅ Command executed successfully');
    console.log('Output:', result.stdout);
    
    // Test Python
    console.log('\n🐍 Testing Python...');
    const pythonResult = await sandbox.commands.run('python --version');
    console.log('✅ Python available');
    console.log('Version:', pythonResult.stdout);
    
    // Test port forwarding
    console.log('\n🌐 Testing port forwarding...');
    const host = sandbox.getHost(8000);
    console.log('✅ Port forwarding works');
    console.log('Host:', host);
    console.log('URL: https://' + host);
    
    console.log('\n🎉 All tests passed!');
    console.log('\n✅ E2B is working correctly!');
    console.log('You can now use it in your application.');
    
  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
    console.error('\nFull error:', error);
    
    if (error.message.includes('403')) {
      console.log('\n💡 Possible solutions:');
      console.log('1. Check your E2B_API_KEY is valid');
      console.log('2. Get a new key from: https://e2b.dev/dashboard');
      console.log('3. Make sure you have credits in your E2B account');
    }
    
    process.exit(1);
  }
}

testE2B();
