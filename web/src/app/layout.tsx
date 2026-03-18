import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "智能影像分割标注",
  description: "专业医学影像分割标注工具 - 支持 117+ 解剖结构自动分割",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full overflow-hidden">
      <body className="antialiased h-full overflow-hidden">
        {children}
      </body>
    </html>
  );
}
