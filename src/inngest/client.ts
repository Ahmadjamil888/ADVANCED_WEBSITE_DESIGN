import { Inngest } from "inngest";

// Create a client to send and receive events for zehanxtech.com
export const inngest = new Inngest({ 
  id: "byte-ai",
  name: "Byte-AI",
  eventKey: process.env.INNGEST_EVENT_KEY,
});