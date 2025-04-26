import os
import random
import textwrap
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import json
import re
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

class MemeMind:
    def __init__(self):
        # Initialize sentiment analysis pipeline - free from Hugging Face
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        # Initialize text generation model - free from Hugging Face
        try:
            # Use a smaller model that works offline
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.text_generator_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_generator_model = AutoModelForCausalLM.from_pretrained(model_name)
            self.has_text_generator = True
        except Exception as e:
            print(f"Could not load text generation model: {e}")
            print("Falling back to rule-based caption generation")
            self.has_text_generator = False

        # Template bank - each template has a sentiment, context tags, and image URL
        self.templates = [
            {
                "name": "Distracted Boyfriend",
                "sentiment": "positive",
                "contexts": ["comparison", "temptation", "choice"],
                "url": "https://i.imgflip.com/1ur9b0.jpg",
                "text_positions": [
                    {"x": 160, "y": 215, "width": 140, "align": "center"},  # Girlfriend
                    {"x": 350, "y": 130, "width": 140, "align": "center"},  # Other woman
                    {"x": 240, "y": 60, "width": 140, "align": "center"}    # Boyfriend
                ]
            },
            {
                "name": "Drake Hotline Bling",
                "sentiment": "positive",
                "contexts": ["comparison", "preference"],
                "url": "https://i.imgflip.com/30b1gx.jpg",
                "text_positions": [
                    {"x": 320, "y": 130, "width": 200, "align": "left"},  # Top panel (rejecting)
                    {"x": 320, "y": 350, "width": 200, "align": "left"}   # Bottom panel (approving)
                ]
            },
            {
                "name": "Two Buttons",
                "sentiment": "negative",
                "contexts": ["choice", "difficulty", "stress"],
                "url": "https://i.imgflip.com/1g8my4.jpg",
                "text_positions": [
                    {"x": 145, "y": 100, "width": 140, "align": "center"},  # Left button
                    {"x": 295, "y": 100, "width": 140, "align": "center"},  # Right button
                    {"x": 215, "y": 280, "width": 200, "align": "center"}   # Person sweating
                ]
            },
            {
                "name": "Change My Mind",
                "sentiment": "negative",
                "contexts": ["debate", "opinion", "challenge"],
                "url": "https://i.imgflip.com/24y43o.jpg",
                "text_positions": [
                    {"x": 250, "y": 250, "width": 300, "align": "center"}  # Sign
                ]
            },
            {
                "name": "Surprised Pikachu",
                "sentiment": "negative",
                "contexts": ["unexpected", "shock", "irony"],
                "url": "https://i.imgflip.com/2kbn1e.jpg",
                "text_positions": [
                    {"x": 215, "y": 50, "width": 400, "align": "center"},  # Top text
                    {"x": 215, "y": 400, "width": 400, "align": "center"}  # Bottom text (optional)
                ]
            }
        ]
        
        # Keywords for context detection
        self.context_keywords = {
            "comparison": ["better", "worse", "vs", "versus", "compare", "instead", "prefer", "alternative"],
            "temptation": ["want", "desire", "tempt", "resist", "crave", "distract"],
            "choice": ["choose", "pick", "select", "option", "decide", "dilemma"],
            "debate": ["argue", "debate", "opinion", "view", "stance", "controversial"],
            "unexpected": ["surprise", "shock", "suddenly", "unexpected", "twist", "plot twist"],
            "challenge": ["challenge", "dare", "prove", "convince", "change"],
            "stress": ["stress", "panic", "anxiety", "pressure", "overwhelm"],
            "irony": ["ironic", "irony", "actually", "funny how", "expected", "thought"],
            "opinion": ["think", "believe", "opinion", "view", "stance", "position"],
            "difficulty": ["hard", "difficult", "struggle", "effort", "impossible", "try"],
            "preference": ["like", "prefer", "favorite", "rather", "instead", "better"]
        }
        
        # Available fonts
        self.fonts = {
            "impact": "impact.ttf",  # Classic meme font
            "arial": "arial.ttf",
            "comic": "comic.ttf"
        }
        
        # Default font if system fonts not available
        self.default_font_size = 36
        self.default_font = None
        
        try:
            # Try to load Impact font (classic meme font)
            self.default_font = ImageFont.truetype("impact.ttf", self.default_font_size)
        except IOError:
            # Fallback to default
            self.default_font = ImageFont.load_default()
    
    def analyze_topic(self, topic_text):
        """Analyze the input text for sentiment, context, and key entities using rule-based approaches"""
        
        # Simple sentiment analysis using Hugging Face
        sentiment_result = self.sentiment_analyzer(topic_text)[0]
        sentiment = sentiment_result['label']
        sentiment_score = sentiment_result['score']
        
        # Extract keywords and context using regex and manual matching
        cleaned_text = topic_text.lower()
        words = re.findall(r'\b\w+\b', cleaned_text)
        
        # Context detection based on keywords
        contexts = []
        for context, keywords in self.context_keywords.items():
            for keyword in keywords:
                if keyword in cleaned_text:
                    contexts.append(context)
                    break
        
        # If no contexts detected, add some defaults
        if not contexts:
            contexts = ["comparison", "unexpected", "opinion"]
        
        # Extract key entities (nouns) - simplified approach
        key_entities = []
        for word in words:
            if len(word) > 3 and word not in ["when", "what", "this", "that", "then", "with", "about"]:
                key_entities.append(word)
        
        # Remove duplicates and limit to 5
        key_entities = list(set(key_entities))[:5]
        
        # Map sentiment to emotion
        emotion_map = {
            "POSITIVE": "humor",
            "NEGATIVE": "frustration" 
        }
        emotion = emotion_map.get(sentiment, "neutral")
        
        # Determine tone based on keywords
        tone = "humorous"  # Default tone for memes
        if any(word in cleaned_text for word in ["sad", "tragic", "unfair", "upset"]):
            tone = "ironic"
        elif any(word in cleaned_text for word in ["shocking", "unbelievable", "crazy"]):
            tone = "dramatic"
        
        # Simple meme potential
        meme_potential = f"A {tone} take on {', '.join(key_entities[:2])}"
        
        # Combine analysis
        full_analysis = {
            "original_text": topic_text,
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "emotion": emotion,
            "contexts": contexts[:3],  # Limit to top 3
            "tone": tone,
            "key_entities": key_entities,
            "meme_potential": meme_potential
        }
        
        return full_analysis
    
    def select_template(self, analysis):
        """Select the best meme template based on topic analysis"""
        
        # Simple scoring system for template matching
        template_scores = []
        
        for template in self.templates:
            score = 0
            
            # Match sentiment
            if template["sentiment"].lower() == analysis["sentiment"].lower():
                score += 2
            
            # Match context
            for context in template["contexts"]:
                if context in analysis["contexts"]:
                    score += 1
            
            template_scores.append((template, score))
        
        # Sort by score and get the best match
        template_scores.sort(key=lambda x: x[1], reverse=True)
        
        # If no good match (all scores are 0), choose randomly
        if all(score == 0 for _, score in template_scores):
            return random.choice(self.templates)
        
        return template_scores[0][0]
    
    def generate_captions_rule_based(self, analysis, template):
        """Generate captions using rule-based approach"""
        template_name = template["name"]
        num_captions = len(template["text_positions"])
        
        # Extract key elements for caption generation
        entities = analysis["key_entities"]
        original_text = analysis["original_text"]
        
        # Default captions for each template
        captions = []
        
        if template_name == "Distracted Boyfriend":
            # Format: [responsible option], [tempting option], [who is distracted]
            captions = [
                f"Using {entities[0] if entities else 'normal approach'}",
                f"Trying {entities[1] if len(entities) > 1 else 'new thing'}",
                f"Me" if "me" in original_text.lower() else f"Everyone"
            ]
            
        elif template_name == "Drake Hotline Bling":
            # Format: [rejecting], [approving]
            reject = f"{entities[0] if entities else 'Normal way'}"
            approve = f"{entities[1] if len(entities) > 1 else 'Better way'}"
            
            # If there's a "vs" or comparison in the text, use that
            if " vs " in original_text.lower():
                parts = original_text.lower().split(" vs ")
                reject = parts[0].strip().title()
                approve = parts[1].strip().title()
                
            captions = [reject, approve]
            
        elif template_name == "Two Buttons":
            # Format: [option 1], [option 2], [who is deciding]
            captions = [
                f"{entities[0] if entities else 'Option 1'}",
                f"{entities[1] if len(entities) > 1 else 'Option 2'}",
                "Me trying to decide"
            ]
            
        elif template_name == "Change My Mind":
            # Format: [controversial opinion]
            opinion = original_text
            if len(opinion) > 50:
                opinion = opinion[:47] + "..."
            captions = [opinion]
            
        elif template_name == "Surprised Pikachu":
            # Format: [action], [reaction]
            captions = [
                f"When {original_text}",
                "Surprised reaction"
            ]
        
        # Ensure we have the right number of captions
        while len(captions) < num_captions:
            captions.append("")
            
        return captions[:num_captions]
    
    def generate_captions_with_model(self, analysis, template):
        """Generate captions using the Hugging Face language model"""
        template_name = template["name"]
        num_captions = len(template["text_positions"])
        
        # Create prompt based on template type
        prompt = f"Write {num_captions} funny captions for a '{template_name}' meme about: {analysis['original_text']}\n\n"
        
        # Add template-specific instructions
        if template_name == "Distracted Boyfriend":
            prompt += """Format:
1. [What should be the normal choice]
2. [The tempting, often wrong choice]
3. [Who is making the choice]"""
        
        elif template_name == "Drake Hotline Bling":
            prompt += """Format:
1. [What Drake is rejecting]
2. [What Drake is preferring/approving]"""
        
        elif template_name == "Two Buttons":
            prompt += """Format:
1. [Left button option]
2. [Right button option]
3. [Who is facing the dilemma]"""
        
        elif template_name == "Change My Mind":
            prompt += """Format:
1. [The controversial opinion]"""
        
        elif template_name == "Surprised Pikachu":
            prompt += """Format:
1. [The setup/action]
2. [The reaction]"""
        
        try:
            # Generate text
            inputs = self.text_generator_tokenizer(prompt, return_tensors="pt")
            output = self.text_generator_model.generate(
                inputs["input_ids"],
                max_length=200,
                temperature=0.7,
                num_return_sequences=1
            )
            
            # Decode the output
            generated_text = self.text_generator_tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract captions using regex
            pattern = r"\d+\.\s*(.+?)(?=\n\d+\.|\n\n|$)"
            matches = re.findall(pattern, generated_text)
            
            captions = [match.strip() for match in matches]
            
            # Ensure we have the right number of captions
            while len(captions) < num_captions:
                captions.append("")
                
            return captions[:num_captions]
            
        except Exception as e:
            print(f"Error generating captions with model: {e}")
            # Fall back to rule-based
            return self.generate_captions_rule_based(analysis, template)
    
    def generate_captions(self, analysis, template):
        """Generate appropriate captions based on the template and topic analysis"""
        if self.has_text_generator:
            return self.generate_captions_with_model(analysis, template)
        else:
            return self.generate_captions_rule_based(analysis, template)
    
    def download_image(self, url):
        """Download image from URL"""
        response = requests.get(url)
        return Image.open(BytesIO(response.content))
    
    def add_text_to_image(self, img, text, position, font=None, font_size=36):
        """Add text to image with proper meme styling"""
        if font is None:
            font = self.default_font
        
        draw = ImageDraw.Draw(img)
        
        # Wrap text to fit width
        wrapped_text = textwrap.fill(text, width=position["width"] // (font_size // 3))
        
        # Calculate text position
        x = position["x"]
        y = position["y"]
        
        # Draw text outline (black)
        for offset_x in range(-2, 3):
            for offset_y in range(-2, 3):
                draw.text((x + offset_x, y + offset_y), wrapped_text, font=font, fill="black", align=position["align"])
        
        # Draw text (white)
        draw.text((x, y), wrapped_text, font=font, fill="white", align=position["align"])
        
        return img
    
    def generate_meme(self, topic_text):
        """Main function to generate a meme from input text"""
        
        # 1. Analyze the topic
        analysis = self.analyze_topic(topic_text)
        print(f"Analysis: {json.dumps(analysis, indent=2)}")
        
        # 2. Select an appropriate template
        template = self.select_template(analysis)
        print(f"Selected template: {template['name']}")
        
        # 3. Generate captions
        captions = self.generate_captions(analysis, template)
        print(f"Generated captions: {captions}")
        
        # 4. Download the template image
        img = self.download_image(template["url"])
        
        # 5. Add captions to the image
        for i, position in enumerate(template["text_positions"]):
            if i < len(captions) and captions[i]:
                img = self.add_text_to_image(
                    img, 
                    captions[i], 
                    position, 
                    font_size=self.default_font_size
                )
        
        # 6. Return the final meme image
        return img

# Example usage
if __name__ == "__main__":
    meme_generator = MemeMind()
    
    # Example topics
    topics = [
        "Tech companies laying off workers while reporting record profits",
        "Students pulling all-nighters before exams",
        "When your code works but you don't know why",
        "The price of concert tickets in 2025"
    ]
    
    # Generate a meme for one of the topics
    topic = random.choice(topics)
    print(f"Generating meme for topic: {topic}")
    
    meme = meme_generator.generate_meme(topic)
    
    # Save the meme
    output_filename = f"meme_{hash(topic) % 10000}.png"
    meme.save(output_filename)
    print(f"Meme saved as {output_filename}")