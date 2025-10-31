import { serve } from "inngest/next";
import { inngest } from "../../../inngest/client";
import { helloWorld, generateModelCode, deployToHuggingFace } from "../../../inngest/functions";

// Create an API that serves zero functions
export const { GET, POST, PUT } = serve({
  client: inngest,
  functions: [
    helloWorld, // <-- This is where you'll always add all your functions
    generateModelCode, // AI model generation function
    deployToHuggingFace, // Hugging Face deployment function
  ],
});