import * as z from "zod";

export const formSchema = z.object({
  prompt: z.instanceof(Buffer),
});
