import { useState, useEffect } from "react";

const prompts = [
  "Create a teaching assistant that helps grade student essays and provides feedback",
  "Build a research assistant that summarizes academic papers and extracts key findings",
  "Design a study guide generator that creates practice questions from course materials",
  "Build a plagiarism checker that analyzes student submissions for academic integrity",
  "Create a lecture note organizer that structures class notes into study materials",
  "Design a citation helper that formats references in APA, MLA, and Chicago styles",
  "Build a course planning assistant that helps professors organize semester schedules",
  "Create a peer review facilitator that guides students through constructive feedback",
  "Design a thesis advisor that helps graduate students structure their research",
  "Build an exam preparation tool that generates practice tests from course content"
];

const TYPING_SPEED = 70;
const DELETING_SPEED = 50;

export function TypewriterMessage() {
  const [currentPromptIndex, setCurrentPromptIndex] = useState(0);
  const [displayedText, setDisplayedText] = useState("");
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    const currentPrompt = prompts[currentPromptIndex];
    const speed = isDeleting ? DELETING_SPEED : TYPING_SPEED;

    const handleTyping = () => {
      if (!isDeleting) {
        if (displayedText.length < currentPrompt.length) {
          setDisplayedText(currentPrompt.substring(0, displayedText.length + 1));
        } else {
          setTimeout(() => setIsDeleting(true), 2000);
        }
      } else {
        if (displayedText.length > 0) {
          setDisplayedText(currentPrompt.substring(0, displayedText.length - 1));
        } else {
          setIsDeleting(false);
          setCurrentPromptIndex((prevIndex) => (prevIndex + 1) % prompts.length);
        }
      }
    };

    const timer = setTimeout(handleTyping, speed);
    return () => clearTimeout(timer);
  }, [displayedText, isDeleting, currentPromptIndex]);

  return (
    <div className="flex justify-end mb-4">
      <div className="max-w-[85%]">
        <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl rounded-tr-sm px-5 py-3 shadow-md">
          <p className="text-white text-[15px] leading-[22px] break-words">
            {displayedText}
            <span className="inline-block w-[2px] h-[18px] bg-white ml-0.5 animate-pulse align-middle" />
          </p>
        </div>
      </div>
    </div>
  );
}



