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
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar } from "react-chartjs-2";
import { faker } from "@faker-js/faker";
import "chart.js/auto";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);
type Chat = {
  content: string;
  role: string;
};

const options = [
  {
    responsive: true,
    plugins: {
      legend: {
        position: "top" as const,
      },
      title: {
        display: true,
        text: "DIET Model",
      },
    },
  },
  {
    responsive: true,
    plugins: {
      legend: {
        position: "top" as const,
      },
      title: {
        display: true,
        text: "Transformer Model",
      },
    },
  },
  {
    responsive: true,
    plugins: {
      legend: {
        position: "top" as const,
      },
      title: {
        display: true,
        text: "SVM Model",
      },
    },
  },
];

const ConversationPage = () => {
  const router = useRouter();
  const [messages, setMessages] = useState<[]>([]);
  const [chartDataArr, setChartDataArr] = useState([]);
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: { prompt: "" },
  });

  const isLoading = form.formState.isSubmitting;

  const getModel = async (url: string, prompt: string, nameModel: string) => {
    const arrayConfidence: any[] = [];
    const arrIntentName: any[] = [];
    const uuid = uuidv4();

    const response = await axios.post(`${url}`, {
      sender: `${uuid}`,
      text: prompt,
    });
    // @ts-ignore

    const responseData = await response.data;
    // console.log(responseData);
    // let text = `text: ${responseData.text} ` + "\n";
    // let text = `Model: ${nameModel} \n\n`;

    await responseData.intent_ranking.map((intent: any) => {
      // text += `- ${intent.name ? intent.name : intent.intent}: ` + "\n\n";

      if (nameModel == "TRANSF") {
        arrIntentName.push(intent.intent);
      } else {
        arrIntentName.push(intent.name);
      }
      const conf = parseFloat(intent.confidence).toFixed(5);
      arrayConfidence.push(conf);
    });

    return {
      labels: arrIntentName,
      datasets: [
        {
          label: "Confidence",
          data: arrayConfidence,
          backgroundColor: "rgba(255, 99, 132, 0.5)",
        },
      ],
    };
  };

  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    try {
      const responsefromTransformer = await getModel(
        `${process.env.NEXT_PUBLIC_API_RASA_URL_TRANSF}`,
        values.prompt,
        "TRANSF"
      );

      const responsefromDIET = await getModel(
        `${process.env.NEXT_PUBLIC_API_RASA_URL_DIET}`,
        values.prompt,
        "DIET"
      );

      const responsefromSVM = await getModel(
        `${process.env.NEXT_PUBLIC_API_RASA_URL_SVM}`,
        values.prompt,
        "SVM"
      );
      const userMessage = {
        role: "user",
        content: [values.prompt],
      };
      setChartDataArr([
        responsefromDIET,
        responsefromTransformer,
        responsefromSVM,
      ]);

      const reply = {
        role: "bot",
        content: [responsefromDIET, responsefromTransformer, responsefromSVM],
        confs: [responsefromTransformer, responsefromDIET, responsefromSVM],
      };
      // console.log(reply);
      // @ts-ignore
      // setResponse1(arrayConfidence1);

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
                <div
                  className={`flex w-full text-base whitespace-pre-line flex-row gap-x-20 ${
                    message.role === "user" && "justify-end"
                  } `}
                >
                  {message.content.map((cont, index) => (
                    <div key={index} className="flex ">
                      {message.role == "user" && <div>{cont}</div>}
                      {/*
                        <div>
                          {message.confs && message.role !== "user" && (
                            <div className="flex flex-col justify-center ml-2 pt-6">
                              {message.confs[index].map((item) => (
                                <p key={index} className="text-base mt-6 ">
                                  {item}
                                </p>
                              ))}
                            </div>
                          )}
                        </div>
                      } */}
                      <div>
                        {message.confs && message.role !== "user" && (
                          <Bar
                            options={options[index]}
                            data={chartDataArr[index]}
                            height={400}
                            width={400}
                          />
                        )}
                      </div>
                    </div>
                  ))}
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
