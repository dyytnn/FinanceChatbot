"use client";

import * as z from "zod";
import Heading from "@/components/heading";
import { Download, ImageIcon } from "lucide-react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { amountOptions, formSchema, resolutionOptions } from "./constants";
import { Form, FormControl, FormField, FormItem } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";
import { useState } from "react";

import axios from "axios";
import Empty from "@/components/empty";
import Loader from "@/components/loader";

import { Card, CardFooter } from "@/components/ui/card";
import Image from "next/image";
import UserAvatar from "@/components/user-avatar";
import BotAvatar from "@/components/bot-avatar";
import { cn } from "@/lib/utils";
type Chat = {
  content: string;
  role: string;
};
const ImagePage = () => {
  const router = useRouter();
  const [messages, setMessages] = useState<Chat[]>([]);

  const [audioTest, setAudioTest] = useState("");

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: { prompt: "", amount: "1", resolution: "512x512" },
  });

  const isLoading = form.formState.isSubmitting;

  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    try {
      setMessages([]);

      const response = await axios.post(
        "https://ntt123-viettts.hf.space/run/predict",
        {
          data: ["Xin Chào"],
        }
      );
      setAudioTest(response.data.data[0].name);
      console.log(audioTest);
      // const urls = response.data.map((image: { url: string }) => image.url);

      // setMessages(urls);
      form.reset();
    } catch (error: any) {
      console.log(error);
      // if (error?.response?.status === 403) {
      //   proModal.onOpen();
      // } else {
      //   toast.error("Something went wrong.");
      // }
    } finally {
      router.refresh();
    }
  };

  return (
    <div>
      <Heading
        title="Speech to Speech"
        description="Voice with us"
        icon={ImageIcon}
        iconColor="text-pink-700"
        bgColor="bg-pink-700/10"
      />
      <div className="px-4 lg:px-8">
        <div className="space-y-4 mt-4">
          {isLoading && (
            <div className="p-20">
              <Loader />
            </div>
          )}
          {messages.length === 0 && !isLoading && (
            <div>
              <Empty label="No conversation generated." />
            </div>
          )}
          <div className="flex flex-col-reverse gap-y-4">
            {
              //{messages.map((message) => (
              // <div
              //   className={cn(
              //     "p-8 w-full flex items-start gap-x-8 rounded-lg",
              //     message.role === "user"
              //       ? "bg-white border border-black/10 flex justify-end items-center"
              //       : "bg-muted items-center"
              //   )}
              //   key={message.content}
              // >
              //   <div className={`${message.role === "user" && "order-2"}`}>
              //     {message.role === "user" ? <UserAvatar /> : <BotAvatar />}
              //   </div>
              //   {/* <p className="text-sm">{message.content}</p> */}
              // </div>
              //))}
            }
            <audio src={audioTest} controls />
          </div>
          <div>
            <Form {...form}>
              <form
                onSubmit={form.handleSubmit(onSubmit)}
                className="rounded-lg border w-full p-4 px-3 md:px-6 focus-within:shadow-sm grid grid-cols-12 gap-2"
              >
                <FormField
                  name="prompt"
                  render={({ field }) => (
                    <FormItem className="col-span-12 lg:col-span-10">
                      <FormControl className="m-0 p-0">
                        <Input
                          className="border-0 outline-none focus-visible:ring-0 focus-visible:ring-transparent"
                          disabled={isLoading}
                          placeholder="A picture for sunset"
                          {...field}
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />

                <Button
                  className="col-span-12 lg:col-span-2 w-full"
                  disabled={isLoading}
                >
                  Send
                </Button>
              </form>
            </Form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImagePage;
