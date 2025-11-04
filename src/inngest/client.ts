import { Inngest } from "inngest";

// Create a client to send and receive events for zehanxtech.com
export const inngest = new Inngest({ 
  id: "zehanx-ai-workspace",
  name: "zehanx AI Workspace",
  eventKey: process.env.INNGEST_EVENT_KEY,
});