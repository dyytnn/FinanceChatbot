import React from "react";

import { UserButton } from "@clerk/nextjs";
import MobileSideBar from "./mobile-sidebar";
import { getApiLimitCount } from "@/lib/api-limit";
interface NavbarProps {
  apiLimitCount?: number;
}
const Navbar: React.FC<NavbarProps> = ({ apiLimitCount }) => {
  return (
    <div className="flex items-center p-4">
      <MobileSideBar apiLimitCount={apiLimitCount} />
      <div className="flex w-full justify-end">
        <UserButton afterSignOutUrl="/" />
      </div>
    </div>
  );
};

export default Navbar;
