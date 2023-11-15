"use client";

import * as z from "zod";
import Heading from "@/components/heading";
import { Mic, Music, Send } from "lucide-react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { formSchema } from "./constants";
import { Form, FormControl, FormField, FormItem } from "@/components/ui/form";

import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import { useReactMediaRecorder } from "react-media-recorder-2";
import SpeechRecognition, {
  useSpeechRecognition,
} from "react-speech-recognition";
import axios from "axios";
import Empty from "@/components/empty";
import Loader from "@/components/loader";

import { cn } from "@/lib/utils";
import UserAvatar from "@/components/user-avatar";
import BotAvatar from "@/components/bot-avatar";

type Chat = {
  content: any;
  role: string;
};
const STTPage = () => {
  const router = useRouter();
  const [messages, setMessages] = useState<Chat[]>([]);
  const [text, setText] = useState("");
  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition,
  } = useSpeechRecognition();
  // const [music, setMusic] = useState<string>();
  const { stopRecording, startRecording, mediaBlobUrl } = useReactMediaRecorder(
    {
      audio: true,
    }
  );
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: { prompt: Buffer.from("") },
  });

  const isLoading = form.formState.isSubmitting;

  useEffect(() => {
    setText(transcript);
  }, [transcript]);

  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    try {
      console.log(transcript);
      console.log(text);

      if (mediaBlobUrl) {
        const uuid = uuidv4();
        const audioBlob = await fetch(mediaBlobUrl).then((r) => r.blob());
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = Buffer.from(arrayBuffer);
        // const audiofile = new File([audioBlob], "audiofile.mp3", {
        //   type: "audio/mpeg",
        // });

        form.setValue("prompt", audioBuffer);
        // const response = await axios.post("/api/music", values, {
        //   headers: {
        //     "Content-Type": "multipart/form-data",
        //   },
        // });

        const userMessage = {
          role: "user",
          content: mediaBlobUrl,
        };

        const response = await axios.post(
          "http://localhost:5005/webhooks/rest/webhook",
          {
            sender: `${uuid}`,
            message: text,
          }
        );

        const reply = {
          role: "bot",
          content: response.data[0].text,
        };
        // const reply = {
        //   role: "bot",
        //   content: "hello",
        // };
        setMessages((current) => [...current, userMessage, reply]);

        form.reset();
      }

      // if (error?.response?.status === 403) {
      //   proModal.onOpen();
      // } else {
      //   toast.error("Something went wrong.");
      // }
      resetTranscript();
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

  const onStart = () => {
    SpeechRecognition.startListening({
      language: "vi-VN",
      continuous: true,
      interimResults: true,
    });
    startRecording();
  };
  const onStop = () => {
    SpeechRecognition.stopListening();
    stopRecording();
  };

  return (
    <div>
      {browserSupportsSpeechRecognition ? (
        <div>
          <Heading
            title="Text to Speech"
            description="Voice with us"
            icon={Music}
            iconColor="text-emerald-500"
            bgColor="bg-emerald-500/10"
          />
          <div className="px-4 lg:px-8 ">
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
              {/* {!music && !isLoading && (
            <div>
              <Empty label="No music generated." />
            </div>
          )}
          {music && (
            <audio controls className="w-full mt-8">
              <source src={music} />
            </audio>
          )} */}
            </div>
            <div className="flex flex-col gap-y-4">
              {messages.map((message, index) => (
                <div
                  className={cn(
                    "p-8 w-full flex items-start gap-x-8 rounded-lg ",
                    message.role === "user"
                      ? "bg-white border border-black/10 flex justify-end items-center "
                      : "bg-muted items-center "
                  )}
                  key={index}
                >
                  <div className={`${message.role === "user" && "order-2"}`}>
                    {message.role === "user" ? <UserAvatar /> : <BotAvatar />}
                  </div>
                  {message.role === "user" ? (
                    <audio src={message.content} controls />
                  ) : (
                    <p className="text-sm">{message.content}</p>
                  )}
                </div>
              ))}
            </div>
            <div>
              <Form {...form}>
                <form
                  onSubmit={form.handleSubmit(onSubmit)}
                  encType="multipart/form-data"
                  className="rounded-lg border w-full p-4 px-3 md:px-6 focus-within:shadow-sm grid grid-cols-[repeat(18,minmax(0,1fr))] gap-2"
                >
                  <FormField
                    name="prompt"
                    render={({ field }) => (
                      <FormItem className="col-[_span_18_/_span_18] lg:col-[_span_16_/_span_16]">
                        <FormControl className="m-0 p-0">
                          <audio
                            src={mediaBlobUrl}
                            controls
                            {...field}
                            className="w-full"
                          />
                        </FormControl>
                      </FormItem>
                    )}
                  />

                  {listening ? (
                    <Button
                      type="button"
                      onClick={onStop}
                      className="col-[_span_18_/_span_18] lg:col-span-1 w-12 h-12 rounded-full ml-7 bg-rose-500 hover:bg-rose-500/80"
                      disabled={isLoading}
                    >
                      <Mic />
                    </Button>
                  ) : (
                    <Button
                      type="button"
                      onClick={onStart}
                      className="col-[_span_18_/_span_18] lg:col-span-1 w-12 h-12 rounded-full ml-7"
                      disabled={isLoading}
                    >
                      <Mic />
                    </Button>
                  )}
                  <Button
                    type="submit"
                    className=" w-12 h-12 rounded-full ml-4"
                    disabled={isLoading}
                  >
                    <Send />
                  </Button>
                </form>
              </Form>
            </div>
          </div>
        </div>
      ) : (
        <div>
          <h1>Your browser has no speech recognition support</h1>
        </div>
      )}
    </div>
    // <div>
    //   <p>{status}</p>
    //   <button onClick={startRecording}>Start Recording</button>
    //   <button onClick={stopRecording}>Stop Recording</button>

    // </div>
  );
};

export default STTPage;
