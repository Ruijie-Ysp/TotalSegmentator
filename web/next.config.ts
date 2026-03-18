import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // 配置 Turbopack 根目录为 web 目录，避免与父目录的 package.json 冲突
  turbopack: {
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
