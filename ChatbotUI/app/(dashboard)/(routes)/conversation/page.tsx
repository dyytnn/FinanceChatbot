// @ts-nocheck
"use client";

import * as z from "zod";
import Heading from "@/components/heading";
import { MessageSquare } from "lucide-react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { formSchema } from "./constants";
import { Form, FormControl, FormField, FormItem } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { ChatCompletionRequestMessage } from "openai";
import axios from "axios";
import Empty from "@/components/empty";
import Loader from "@/components/loader";
import { cn } from "@/lib/utils";
import UserAvatar from "@/components/user-avatar";
import BotAvatar from "@/components/bot-avatar";
import { v4 as uuidv4 } from "uuid";
type Chat = {
  content: string;
  role: string;
};

const ConversationPage = () => {
  const router = useRouter();
  const [messages, setMessages] = useState<[]>([]);
  const [response1, setResponse1] = useState([]);
  const [confarray, setConfArray] = useState([]);
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: { prompt: "" },
  });

  const isLoading = form.formState.isSubmitting;
  const arrayConfidence1: any[] = [];

  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    try {
      const uuid = uuidv4();
      const userMessage = {
        role: "user",
        content: values.prompt,
      };

      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_API_RASA_URL}`,
        {
          sender: `${uuid}`,
          text: values.prompt,
        }
      );
      // @ts-ignore

      const responseData = await response.data;
      // console.log(responseData);
      // let text = `text: ${responseData.text} ` + "\n";
      let text = "";
      await responseData.intent_ranking.map((intent: any) => {
        text += `- ${intent.name}: ` + "\n";
        const conf = parseFloat(intent.confidence).toFixed(5);
        arrayConfidence1.push(conf);
      });

      const reply = {
        role: "bot",
        content: text,
        confs: arrayConfidence1,
      };
      // @ts-ignore
      setResponse1(arrayConfidence1);

      setMessages((current) => [...current, userMessage, reply]);

      form.reset();
      // console.log(response1);
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
        title="Conversation"
        description="Our most advanced conversation model."
        icon={MessageSquare}
        iconColor="text-violet-500"
        bgColor="bg-violet-500/10"
      />
      <div className="px-4 lg:px-8">
        <div className="space-y-4 mt-4">
          {isLoading && (
            <div className="p-8 rounded-lg w-full flex items-center justify-center bg-muted">
              <Loader />
            </div>
          )}
          {messages.length === 0 && !isLoading && (
            <div>
              <Empty label="No conversation started." />
            </div>
          )}
          <div className="flex flex-col gap-y-4">
            {messages.map((message, index) => (
              <div
                className={cn(
                  "p-8 w-full h-full flex items-start gap-x-8 rounded-lg ",
                  message.role === "user"
                    ? "bg-white border border-black/10 flex justify-end items-center "
                    : "bg-muted items-center "
                )}
                key={index}
              >
                <div className={`${message.role === "user" && "order-2"}`}>
                  {message.role === "user" ? <UserAvatar /> : <BotAvatar />}
                </div>

                <div className="flex text-base whitespace-pre-line">
                  {message.content}
                  <div className="flex flex-col justify-center ml-2">
                    {message.confs &&
                      message.role !== "user" &&
                      message.confs.map((conf, index) => (
                        <p key={index} className="text-base">
                          {conf}
                        </p>
                      ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div className="sticky bottom-2">
            <Form {...form}>
              <form
                onSubmit={form.handleSubmit(onSubmit)}
                className="rounded-lg border w-full p-4 px-3 md:px-6 focus-within:shadow-sm grid grid-cols-12 gap-2 bg-white "
              >
                <FormField
                  name="prompt"
                  render={({ field }) => (
                    <FormItem className="col-span-12 lg:col-span-10">
                      <FormControl className="m-0 p-0">
                        <Input
                          className="border-0 outline-none focus-visible:ring-0 focus-visible:ring-transparent"
                          disabled={isLoading}
                          placeholder="What are my repayment options ?"
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

export default ConversationPage;
