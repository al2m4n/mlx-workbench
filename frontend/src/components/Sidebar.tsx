import { NavLink } from "react-router-dom";
import {
  MessageSquare,
  Boxes,
  Mic,
  Video,
  LineChart,
  type LucideIcon,
} from "lucide-react";
import clsx from "clsx";

type Item = { to: string; label: string; icon: LucideIcon };

const items: Item[] = [
  { to: "/chat", label: "Chat", icon: MessageSquare },
  { to: "/models", label: "Models", icon: Boxes },
  { to: "/metrics", label: "Metrics", icon: LineChart },
  { to: "/audio", label: "Audio", icon: Mic },
  { to: "/video", label: "Vision", icon: Video },
];

export default function Sidebar() {
  return (
    <aside className="w-56 shrink-0 border-r border-ink-800 bg-ink-900 flex flex-col">
      <div className="px-4 py-5 border-b border-ink-800">
        <div className="font-mono text-sm tracking-tight text-ink-100">
          mlx-workbench
        </div>
        <div className="font-mono text-[10px] text-ink-400 mt-0.5">
          v0.1.0
        </div>
      </div>
      <nav className="flex-1 p-2 space-y-0.5">
        {items.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              clsx(
                "flex items-center gap-2.5 px-3 py-2 rounded-md text-sm transition",
                isActive
                  ? "bg-ink-800 text-ink-100"
                  : "text-ink-300 hover:bg-ink-800/60 hover:text-ink-100",
              )
            }
          >
            <Icon size={16} />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
