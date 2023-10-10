import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

const features = [
  {
    name: "Text to Text",
    title: "Text to Text",
    description: "ChatBot Text to Text Feature",
  },
  {
    name: "Speech to Text",
    title: "Text to Text",
    description: "VoiceBot Speech to Text Feature",
  },
  {
    name: "Speech to Speech",
    title: "Speech to Speech",
    description: "VoiceBot Speech to Speech Feature",
  },
];

const LandingContent = () => {
  return (
    <div className="px-10 pb-20">
      <h2 className="text-center text-4xl text-white font-extrabold mb-10">
        Feature
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {features.map((item) => (
          <Link key={item.name} href="#">
            <Card
              key={item.name}
              className="bg-[#192339] border-none text-white"
            >
              <CardHeader>
                <CardTitle className="flex items-center gap-x-2">
                  <div>
                    <p className="text-lg">{item.name}</p>
                    <p className="text-zinc-400 text-sm">{item.title}</p>
                  </div>
                </CardTitle>
                <CardContent className="pt-4 px-0">
                  {item.description}
                </CardContent>
              </CardHeader>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
};

export default LandingContent;
