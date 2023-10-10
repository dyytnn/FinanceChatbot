import { checkApiLimit, increaseApiLimit } from "@/lib/api-limit";
import { auth } from "@clerk/nextjs";
import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const { userId } = auth();
    const body = await req.formData();
    // const { prompt } = body;

    console.log(body);
    if (!userId) {
      return new NextResponse("Unauthorized", { status: 401 });
    }

    // const freeTrial = await checkApiLimit();

    // if (!freeTrial) {
    //   return new NextResponse("Free trial has expired.", { status: 403 });
    // }

    return NextResponse.json("Hello");
  } catch (error) {
    console.log("[MUSIC_ERROR]", error);
    return new NextResponse("Internal Error", { status: 500 });
  }
}
