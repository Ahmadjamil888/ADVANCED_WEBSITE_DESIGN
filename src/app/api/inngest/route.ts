import { serve } from "inngest/next";
import { inngest } from "../../../inngest/client";
import { generateAIModel, deployToHuggingFace } from "../../../inngest/functions";

// Create an API that serves all AI functions
export const { GET, POST, PUT } = serve({
  client: inngest,
  functions: [
    generateAIModel, // Complete AI model generation function
    deployToHuggingFace, // Hugging Face deployment function
  ],
});