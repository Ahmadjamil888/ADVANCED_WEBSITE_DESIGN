import { serve } from "inngest/next";
import { inngest } from "@/lib/inngest";
import { 
  trainAIModel, 
  deployToHuggingFace, 
  findDataset,
  generateModelCode 
} from "@/lib/inngest/functions";

// Create the Inngest serve handler
export const { GET, POST, PUT } = serve({
  client: inngest,
  functions: [
    trainAIModel,
    deployToHuggingFace,
    findDataset,
    generateModelCode
  ],
});