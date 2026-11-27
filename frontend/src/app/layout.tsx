import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Azure AI & Qdrant RAG Chat",
  description: "Chat UI connected to FastAPI + Azure OpenAI + Azure AI Search + Qdrant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
