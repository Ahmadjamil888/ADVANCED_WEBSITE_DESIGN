-- E2B AI Workspace Schema
-- Based on zehanxtech/vibe project structure

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create ENUM types
CREATE TYPE "MessageRole" AS ENUM ('USER', 'ASSISTANT');
CREATE TYPE "MessageType" AS ENUM ('RESULT', 'ERROR');

-- Projects table (equivalent to chats/workspaces)
CREATE TABLE "Project" (
  "id" TEXT PRIMARY KEY DEFAULT uuid_generate_v4()::TEXT,
  "name" TEXT NOT NULL DEFAULT 'Untitled Project',
  "userId" TEXT NOT NULL,
  "createdAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updatedAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Messages table
CREATE TABLE "Message" (
  "id" TEXT PRIMARY KEY DEFAULT uuid_generate_v4()::TEXT,
  "content" TEXT NOT NULL,
  "role" "MessageRole" NOT NULL,
  "type" "MessageType" NOT NULL,
  "projectId" TEXT,
  "createdAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updatedAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY ("projectId") REFERENCES "Project"("id") ON DELETE CASCADE
);

-- Fragments table (sandbox results with files)
CREATE TABLE "Fragment" (
  "id" TEXT PRIMARY KEY DEFAULT uuid_generate_v4()::TEXT,
  "messageId" TEXT NOT NULL UNIQUE,
  "sandboxUrl" TEXT NOT NULL,
  "title" TEXT NOT NULL,
  "files" JSONB NOT NULL,
  "createdAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updatedAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY ("messageId") REFERENCES "Message"("id") ON DELETE CASCADE
);

-- Usage tracking table
CREATE TABLE "Usage" (
  "key" TEXT PRIMARY KEY,
  "points" INTEGER NOT NULL DEFAULT 0,
  "expire" TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX "idx_project_userId" ON "Project"("userId");
CREATE INDEX "idx_message_projectId" ON "Message"("projectId");
CREATE INDEX "idx_message_createdAt" ON "Message"("createdAt" DESC);
CREATE INDEX "idx_fragment_messageId" ON "Fragment"("messageId");

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW."updatedAt" = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add triggers for updated_at
CREATE TRIGGER update_project_updated_at BEFORE UPDATE ON "Project"
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_message_updated_at BEFORE UPDATE ON "Message"
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_fragment_updated_at BEFORE UPDATE ON "Fragment"
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Sample data (optional)
-- INSERT INTO "Project" ("id", "name", "userId") VALUES 
-- ('sample-project-1', 'My First AI Model', 'user-123');
