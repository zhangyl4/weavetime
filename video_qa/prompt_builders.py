"""
Prompt builders for different question formats.
This module provides a modular way to construct different prompt templates
for streamingbench and other QA tasks.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class PromptBuilder(ABC):
    """Base class for prompt builders."""
    
    @abstractmethod
    def build_prompt(self, question: str, candidates: List[str], **kwargs) -> Dict[str, Any]:
        """
        Build a prompt from question and candidates.
        
        Args:
            question: The question string
            candidates: List of candidate answers
            **kwargs: Additional metadata (e.g., task, start_time, end_time)
            
        Returns:
            dict with keys:
                - question: Original question
                - formatted_question: Formatted question text
                - prompt: Final prompt for the model
        """
        pass


class DefaultMCQAPromptBuilder(PromptBuilder):
    """Default multiple-choice QA prompt builder."""
    
    def __init__(self, choice_letters: List[str] = None):
        self.choice_letters = choice_letters or ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    def build_prompt(self, question: str, candidates: List[str], **kwargs) -> Dict[str, Any]:
        """Build default MCQ prompt: Question + Options + Instruction"""
        formatted_choices = "\n".join([
            "(" + self.choice_letters[i] + ") " + candidate 
            for i, candidate in enumerate(candidates)
        ])
        formatted_question = f"Question: {question}\nOptions:\n{formatted_choices}\nOnly give the best option."
        
        return {
            "question": question,
            "formatted_question": formatted_question,
            "prompt": None  # Will be set by the caller using qa_model.get_prompt()
        }


PROMPT_TEMPLATE = '''You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {}

Options:
{}
{}
{}
{}'''

PROMPT_TEMPLATE_WITHOUT_OPTIONS = '''You are an advanced video question-answering AI assistant. You have been provided with a video and a question related to the video. Your task is to carefully analyze the video and provide the answer to the question. 

Question: {}
'''

class StreamingBenchPromptBuilder(PromptBuilder):
    """Prompt builder for StreamingBench tasks with task-specific formatting."""
    
    def __init__(self, choice_letters: List[str] = None, model: str = 'llava_ov_7b'):
        self.choice_letters = choice_letters or ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.model = model
    
    def build_prompt(self, question: str, candidates: List[str], **kwargs) -> Dict[str, Any]:
        """
        Build StreamingBench prompt with optional task information.
        
        Args:
            question: The question string
            candidates: List of candidate answers
        """
        if len(candidates) > 0:
            if not candidates[0].startswith("A."):
                candidates = [f"A. {candidates[0]}", f"B. {candidates[1]}", f"C. {candidates[2]}", f"D. {candidates[3]}"]

            formatted_question = PROMPT_TEMPLATE.format(question, *candidates)
            formatted_question += "\n\nOnly give the best option."
        else:
            formatted_question = PROMPT_TEMPLATE_WITHOUT_OPTIONS.format(question)
            formatted_question += "\n\nAnswer:"
        

        if self.model == 'qwen2vl_7b':
            prompt = f"<|vision_end|>\n{formatted_question}<|im_end|><|im_start|>assistant\n"
            prompt += 'The best option is:'
        else:
            prompt =  f"\n{formatted_question}<|im_end|><|im_start|>assistant\n"
            prompt += 'The best option is:'
        
        return {
            "question": question,
            "formatted_question": formatted_question,
            "prompt": prompt 
        }


class CompactPromptBuilder(PromptBuilder):
    """Compact prompt builder with minimal formatting."""
    
    def __init__(self, choice_letters: List[str] = None):
        self.choice_letters = choice_letters or ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    def build_prompt(self, question: str, candidates: List[str], **kwargs) -> Dict[str, Any]:
        """Build compact prompt: Question + Options (no extra instructions)"""
        formatted_choices = "\n".join([
            self.choice_letters[i] + ". " + candidate 
            for i, candidate in enumerate(candidates)
        ])
        formatted_question = f"{question}\n{formatted_choices}"
        
        return {
            "question": question,
            "formatted_question": formatted_question,
            "prompt": None
        }


class InstructionRichPromptBuilder(PromptBuilder):
    """Prompt builder with rich instructions."""
    
    def __init__(self, choice_letters: List[str] = None, instruction: str = None):
        self.choice_letters = choice_letters or ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.instruction = instruction or "Please carefully analyze the video and select the best answer from the options below."
    
    def build_prompt(self, question: str, candidates: List[str], **kwargs) -> Dict[str, Any]:
        """Build prompt with rich instructions"""
        formatted_choices = "\n".join([
            "(" + self.choice_letters[i] + ") " + candidate 
            for i, candidate in enumerate(candidates)
        ])
        formatted_question = f"{self.instruction}\n\nQuestion: {question}\n\nOptions:\n{formatted_choices}\n\nPlease provide your answer as a single letter (A, B, C, or D)."
        
        return {
            "question": question,
            "formatted_question": formatted_question,
            "prompt": None
        }


class NumberedPromptBuilder(PromptBuilder):
    """Prompt builder that uses numbers instead of letters."""
    
    def build_prompt(self, question: str, candidates: List[str], **kwargs) -> Dict[str, Any]:
        """Build prompt with numbered options (1, 2, 3, ...)"""
        formatted_choices = "\n".join([
            f"{i+1}. {candidate}" 
            for i, candidate in enumerate(candidates)
        ])
        formatted_question = f"Question: {question}\nOptions:\n{formatted_choices}\nAnswer with the option number (1, 2, 3, or 4)."
        
        return {
            "question": question,
            "formatted_question": formatted_question,
            "prompt": None
        }


# Registry for prompt builders
PROMPT_BUILDER_REGISTRY = {
    'default': DefaultMCQAPromptBuilder,
    'streamingbench': StreamingBenchPromptBuilder,
    'compact': CompactPromptBuilder,
    'instruction_rich': InstructionRichPromptBuilder,
    'numbered': NumberedPromptBuilder,
}


def get_prompt_builder(builder_type: str = 'default', **builder_kwargs) -> PromptBuilder:
    """
    Get a prompt builder instance by type.
    
    Args:
        builder_type: Type of prompt builder ('default', 'streamingbench', etc.)
        **builder_kwargs: Additional arguments for the builder constructor
        
    Returns:
        PromptBuilder instance
        
    Examples:
        >>> builder = get_prompt_builder('default')
        >>> builder = get_prompt_builder('streamingbench', include_task_info=True)
        >>> builder = get_prompt_builder('instruction_rich', instruction='Custom instruction')
    """
    if builder_type not in PROMPT_BUILDER_REGISTRY:
        raise ValueError(f"Unknown prompt builder type: {builder_type}. "
                         f"Available types: {list(PROMPT_BUILDER_REGISTRY.keys())}")
    
    builder_class = PROMPT_BUILDER_REGISTRY[builder_type]
    return builder_class(**builder_kwargs)

