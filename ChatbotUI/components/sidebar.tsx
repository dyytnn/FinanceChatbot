"use client";

import { cn } from "@/lib/utils";
import {
  Code,
  FileAudio2Icon,
  FileAudioIcon,
  ImageIcon,
  LayoutDashboard,
  MessageSquare,
  Music,
  Settings,
  VideoIcon,
} from "lucide-react";
import { Montserrat } from "next/font/google";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import FreeCounter from "@/components/free-counter";

const monsterrat = Montserrat({ weight: "600", subsets: ["latin"] });

const routes = [
  {
    label: "Dashboard",
    icon: LayoutDashboard,
    href: "/dashboard",
    color: "text-sky-500",
  },
  {
    label: "Conversation",
    icon: MessageSquare,
    href: "/conversation",
    color: "text-violet-500",
  },
  {
    label: "Text To Speech",
    icon: Music,
    href: "/texttospeech",
    color: "text-emerald-500",
  },

  {
    label: "Speech To Text",
    icon: FileAudio2Icon,
    href: "/speechtotext",
    color: "text-green-700",
  },

  {
    label: "Speech To Speech",
    icon: FileAudioIcon,
    href: "/speechtospeech",
    color: "text-pink-700",
  },

  // {
  //   label: "Settings",
  //   icon: Settings,
  //   href: "/settings",
  // },
];

interface SidebarProps {
  apiLimitCount?: number;
}

const SideBar: React.FC<SidebarProps> = ({ apiLimitCount = 0 }) => {
  const pathName = usePathname();
  return (
    <div className="space-y-4 py-4 flex flex-col h-full bg-[#111827] text-white w-full">
      <div className="px-3 py-2 flex-1">
        <Link href="/dashboard" className="flex items-center pl-3 mb-14">
          <div className="relative w-10 h-10 mr-4">
            <Image fill alt="logo" src="/Artboard_2.png" />
          </div>
          <h1 className={cn("text-2xl font-bold", monsterrat.className)}>
            Athena
          </h1>
        </Link>
        <div className="space-y-1">
          {routes.map((route) => (
            <Link
              href={route.href}
              key={route.href}
              className={cn(
                "text-sm group flex p-3 w-full justify-start font-medium cursor-pointer hover:text-white hover:bg-white/10 rounded-lg transition",
                pathName === route.href
                  ? "text-white bg-white/10 "
                  : "text-zinc-400"
              )}
            >
              <div className="flex items-center flex-1">
                <route.icon className={cn("h-5 w-5 mr-3", route.color)} />
                {route.label}
              </div>
            </Link>
          ))}
        </div>
      </div>
      {/* <FreeCounter apiLimitCount={apiLimitCount} /> */}
    </div>
  );
};

export default SideBar;
