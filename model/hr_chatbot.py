import os
import logging
import requests
import spacy
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from dotenv import load_dotenv
from jsonschema import validate
import pytz
from enum import Enum
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import pickle
from pathlib import Path

# Enhanced environment loading
env_path = Path("D:/Dbms Models/expert_api.env")
load_dotenv(env_path)

# Configure advanced logging
logging.basicConfig(
    filename='hr_negotiations.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.debug("This is a debug message")
logging.info("This is an info message")

# NLP models
class NLPTools:
    def __init__(self):
        self.nlp = self._load_spacy_model()
        self.sentiment_model = self._load_sentiment_model()
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def _load_spacy_model(self):
        try:
            nlp = spacy.load("en_core_web_lg")
            return nlp
        except OSError:
            from spacy.cli import download
            download("en_core_web_lg")
            nlp = spacy.load("en_core_web_lg")
            return nlp

    def _load_sentiment_model(self):
        return lambda text: np.random.uniform(-1, 1)

nlp_tools = NLPTools()

# Constants
class Constants:
    SALARY_API_URL = "https://job-salary-data.p.rapidapi.com/job-salary"
    LLAMA_API_URL = "http://localhost:11434/api/generate"
    MAX_CONCESSION_STEPS = 8
    MAX_INCREASE_PCT = 0.10
    MIN_NEGOTIATION_WORDS = 8
    CACHE_DIR = Path("negotiation_cache")
    BENEFIT_KEYWORDS = {
        'training': ['course', 'training', 'certification', 'upskill', 'development'],
        'mentorship': ['mentor', 'guide', 'advisor', 'coaching'],
        'flexibility': ['remote', 'hybrid', 'flexible', 'work from home'],
        'financial': ['bonus', 'stock', 'equity', 'allowance', 'signing bonus'],
        'wellbeing': ['health', 'wellness', 'gym', 'vacation', 'time off']
    }
    LOW_SALARY_THRESHOLD_PCT = 0.85

# JSON Schema
REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "structured_data": {
            "type": "object",
            "properties": {
                "candidate_overview": {"type": "object"},
                "skills": {"type": "object"},
                "predicted_roles": {"type": "array"}
            },
            "required": ["candidate_overview", "skills", "predicted_roles"]
        },
        "weakness_analysis": {"type": "object"},
        "hr_strategies": {"type": "object"}
    },
    "required": ["structured_data", "weakness_analysis", "hr_strategies"]
}

# Enums
class NegotiationState(Enum):
    INITIAL = 1
    COUNTER_OFFER = 2
    EXPLORING_BENEFITS = 3
    FINAL_OFFER = 4
    CLOSED = 5

class CandidateProfile:
    def __init__(self, report_path: str):
        self.report = self._load_and_validate_report(report_path)
        self.structured_data = self.report['structured_data']
        self.weakness_analysis = self.report['weakness_analysis']
        self.hr_strategies = self.report['hr_strategies']
        self._cache = {}

    def _load_and_validate_report(self, path: str) -> Dict:
        with open(path, 'r') as f:
            report = json.load(f)
            validate(instance=report, schema=REPORT_SCHEMA)
            return report

    def get_name(self) -> str:
        return self.structured_data['candidate_overview'].get('name', 'Candidate')

    def get_experience(self) -> float:
        work_experience = self.structured_data['candidate_overview'].get('work_experience', [])
        total_years = 0.0
        for exp in work_experience:
            if 'date_range' in exp:
                try:
                    start_str, end_str = exp['date_range'].split('–')
                    start_date = datetime.strptime(start_str.strip(), '%b %Y')
                    end_date = datetime.strptime(end_str.strip(), '%b %Y') if end_str.strip().lower() != 'present' else datetime.now()
                    total_years += (end_date - start_date).days / 365.25
                except Exception as e:
                    logging.error(f"Error parsing date range: {e}")
                    continue
        return total_years

    def get_skills_matrix(self) -> Dict[str, float]:
        return {skill: 1.0 for skill in self.structured_data['skills'].get('technical', [])}

    def get_weakness_factor(self) -> float:
        weaknesses = self.weakness_analysis.get('weaknesses', [])
        if not weaknesses:
            return 0.0
        return min(1.0, len(weaknesses) * 0.1)

    def get_roles(self) -> List[str]:
        return self.structured_data.get('predicted_roles', [])

class MarketAnalyzer:
    def __init__(self):
        self.api_key = os.getenv("RAPIDAPI_KEY")
        self.__init__cache()

    def __init__cache(self):
        Constants.CACHE_DIR.mkdir(exist_ok=True)
        return True

    def _get_cache_key(self, role: str) -> str:
        return hashlib.md5(role.encode()).hexdigest() + '.pkl'

    def get_benchmarks(self, role: str) -> Dict:
        cache_file = Constants.CACHE_DIR / self._get_cache_key(role)

        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                if datetime.now().timestamp() - cached_data['timestamp'] < 86400:
                    return cached_data['data']
                
        url = "https://job-salary-data.p.rapidapi.com/job-salary"
        params = {
            "job_title": role,
            "location": "Bangalore",
            "period": "yearly",
            "currency": "INR"
        }

        headers = {
            "X-RapidAPI-Key": "be3db1e9a4msh70342622431a81ep1981cejsnd4fe00bb7d53",
            "X-RapidAPI-Host": "job-salary-data.p.rapidapi.com"
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            if 'data' in data and data['data']:
                info = data['data'][0]
                benchmarks = {
                    'min': info.get('min_base_salary', 1000000) / 100000,
                    'avg': info.get('median_base_salary', 1500000) / 100000,
                    'max': info.get('max_base_salary', 2200000) / 100000,
                    'currency': 'LPA',
                    'updated': datetime.now().isoformat()
                }

                with open(cache_file, 'wb') as f:
                    pickle.dump({'data': benchmarks, 'timestamp': datetime.now().timestamp()}, f)

                return benchmarks
            else:
                return {}

        except Exception as e:
            return {
                'min': 450000,
                'avg': 720000,
                'max': 1200000,
                'currency': 'LPA',
                'updated': datetime.now().isoformat(),
                'source': 'fallback'
            }

class LlamaNegotiationExpert:
    def __init__(self):
        self._test_connection()
        self.context_window = []
        self.max_context = 8
        self.failure_count = 0
        
    def _test_connection(self):
        try:
            response = requests.get("http://localhost:11434")
            if response.status_code != 200:
                raise ConnectionError("Llama server not responding")
            return True
        except Exception as e:
            logging.critical(f"Llama server connection failed: {str(e)}")
            raise
        
    def _generate_prompt(self, negotiation_context: Dict) -> str:
        templates = {
            'initial': (
                "The candidate {candidate} is being offered the position of {role}. "
                "Their profile suggests {strengths}. The initial salary offer is {offer} {currency}, "
                "while the market range is {market_min}-{market_max} {currency}. "
                "Generate the HR's next response to the candidate's initial reaction: '{candidate_reaction}'."
                "Focus on justifying the offer, highlighting benefits, and being open to further discussion."
            ),
            'counter': (
                "The candidate {candidate} has countered with a request of {requested_amount} {currency} "
                "and mentioned interest in benefits like {benefit_requests_str}. "
                "The current offer is {offer} {currency}. The market range is {market_min}-{market_max} {currency}. "
                "Consider the candidate's sentiment ({sentiment}), bluff probability ({bluff_prob}). "
                "Generate the HR's next counter-offer and response, aiming for a mutually agreeable solution. "
                "Be strategic about salary and benefits."
                "have a mind in the previous offer provided and give responses ."
            ),
            'exploring_benefits': (
                "The candidate {candidate} is showing interest in benefits like {benefit_requests_str}. "
                "The current salary offer is {offer} {currency}. "
                "Elaborate the details of the request benefit."
                "Suggest specific and attractive benefit options related to their interests, "
                "and indicate potential flexibility on salary if benefits are accepted."
            ),
            'final': (
                "This is the final offer: {offer} {currency} including {final_benefits}. "
                "Clearly state this is the final offer and request a decision (accept or decline)."
            )
        }

        context_data = {
            'candidate': negotiation_context.get('candidate', 'Candidate'),
            'role': negotiation_context.get('role', 'the position'),
            'offer': negotiation_context.get('current_offer', 'an amount'),
            'currency': negotiation_context.get('currency', 'INR'),
            'market_min': negotiation_context.get('market_min', 'a value'),
            'market_max': negotiation_context.get('market_max', 'a value'),
            'benefits': negotiation_context.get('initial_benefits_str', 'standard benefits'),
            'candidate_reaction': negotiation_context.get('candidate_input', ''),
            'requested_amount': negotiation_context.get('requested_amount', 'an amount'),
            'benefit_requests_str': ", ".join(sum(negotiation_context.get('benefit_requests', {}).values(), [])),
            'sentiment': negotiation_context.get('sentiment', 'neutral'),
            'bluff_prob': negotiation_context.get('bluff_prob', 0.5),
            'final_benefits': negotiation_context.get('offered_benefits_str', 'the agreed benefits'),
            'strengths': "their experience and skills"
        }

        if not self.context_window:
            return templates['initial'].format(**context_data)
        elif negotiation_context.get('state') == NegotiationState.FINAL_OFFER:
            return templates['final'].format(**context_data)
        elif negotiation_context.get('state') == NegotiationState.EXPLORING_BENEFITS:
            return templates['exploring_benefits'].format(**context_data)
        else:
            return templates['counter'].format(**context_data)

    def analyze_conversation(self, negotiation_context: Dict) -> Dict:
        prompt = self._generate_prompt(negotiation_context)
        logging.info(f"Llama API Prompt: {prompt}")

        try:
            response = requests.post(
                Constants.LLAMA_API_URL,
                json={
                    "model": "llama3.1",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.6,
                    }
                },
            )
            response.raise_for_status()
            self.failure_count = 0
            llama_response = response.json().get('response', '')
            logging.info(f"Llama API Raw Response: {llama_response}")
            return {"response": {"text": llama_response}}

        except requests.exceptions.RequestException as e:
            logging.error(f"Llama API Request Error: {e}", exc_info=True)
            self.failure_count += 1
            return self._generate_fallback_response(negotiation_context)
        except json.JSONDecodeError as e:
            logging.error(f"Llama API JSON Decode Error: {e}", exc_info=True)
            self.failure_count += 1
            return self._generate_fallback_response(negotiation_context)
        except Exception as e:
            logging.error(f"Llama API General Error: {e}", exc_info=True)
            self.failure_count += 1
            return self._generate_fallback_response(negotiation_context)

    def _generate_fallback_response(self, context: Dict) -> Dict:
        fallback_responses = [
            "We understand your perspective. Let's see how we can bridge this gap.",
            "Thank you for your feedback. We're reviewing our offer.",
            "We appreciate your interest. Let's explore the complete compensation package.",
            f"Considering your experience, we can slightly increase the offer to {round(context.get('current_offer', 0) * 1.02, 2)} {context.get('currency', 'INR')}."
        ]
        return {"response": {"text": np.random.choice(fallback_responses)}}

class NegotiationEngine:
    def __init__(self, candidate: CandidateProfile, role: str):
        self.candidate = candidate
        self.role = role
        self.market = MarketAnalyzer()
        self.llama = LlamaNegotiationExpert()
        self.benchmarks = self.market.get_benchmarks(role)
        self.state = NegotiationState.INITIAL
        self.offer_history = []
        self.concession_count = 0
        self.current_offer = self._calculate_initial_offer()
        self.offer_history.append(self.current_offer)
        self.offered_benefits = {}
        self.conversation_history = []
        self.candidate_request_history = []
        self.hr_offer_history = []
        self.salary_finalized = False
        
    def _detect_negotiation_intent(self, text: str) -> bool:
        if self.salary_finalized:
            return False 
        
        nlp = spacy.load("en_core_web_lg")
        doc = nlp(text.lower())
        
        acceptance_phrases = {'i accept', 'happy to accept', 'i’m excited to join', 'looking forward to getting started'}
        if any(phrase in text.lower() for phrase in acceptance_phrases):
            return False

        explicit_keywords = {'increase', 'salary', 'compensation', 'counter', 'offer', 
                            'negotiate', 'adjust', 'revise', 'higher'}
        if any(token.lemma_ in explicit_keywords for token in doc):
            return True

        implicit_phrases = {'not enough', 'too low', 'below market', 'reconsider',
                            'bridge gap', 'improve offer', 'better package'}
        if any(phrase in text.lower() for phrase in implicit_phrases):
            return True

        money_ents = [ent for ent in doc.ents if ent.label_ == "MONEY"]
        if len(money_ents) >= 2:
            return True

        comparative_keywords = {"more", "less", "higher", "lower", "above", "below"}
        for token in doc:
            if token.text in comparative_keywords and token.dep_ == "advmod":
                return True

        return False

    def _detect_acceptance(self, text: str) -> bool:
        acceptance_phrases = {
            'accept the offer', 'happy to confirm', 'ready to join', 'pleased to accept',
            'agree to the terms', 'willing to proceed', 'excited to join', 'confirming my acceptance',
            'looking forward to joining', 'eager to start', 'glad to accept', 'happy to accept'
        }

        text_lower = text.lower()

        if any(phrase in text_lower for phrase in acceptance_phrases):
            return True

        doc = nlp_tools.nlp(text_lower)

        acceptance_verbs = {'accept', 'agree', 'confirm', 'proceed', 'join', 'start'}
        for token in doc:
            if token.lemma_ in acceptance_verbs and token.pos_ == 'VERB':
                return True

        enthusiasm_words = {'excited', 'thrilled', 'eager', 'pleased'}
        for token in doc:
            if token.lemma_ in enthusiasm_words and token.pos_ == 'ADJ':
                return True

        return False
    
    def _detect_reject(self, text: str) -> bool:
        rejection_phrases = {
            'decline the offer', 'not interested', 'unable to accept', 'reject the offer',
            'withdraw my application', 'pursue other opportunities', 'not a good fit',
            'cannot accept', 'look elsewhere', 'turn down the offer', 'regret to inform'
        }

        text_lower = text.lower()

        if any(phrase in text_lower for phrase in rejection_phrases):
            return True

        doc = nlp_tools.nlp(text_lower)

        rejection_verbs = {'decline', 'refuse', 'reject', 'withdraw', 'deny'}
        for token in doc:
            if token.lemma_ in rejection_verbs and token.pos_ == 'VERB':
                if any(child.dep_ == 'neg' for child in token.children):
                    return True

        negative_words = {'disappointed', 'unhappy', 'unsatisfactory', 'insufficient', 'unfortunately'}
        if any(token.lemma_ in negative_words and token.pos_ in ('ADJ', 'ADV') for token in doc):
            return True

        money_ents = [ent.text for ent in doc.ents if ent.label_ == "MONEY"]
        if len(money_ents) >= 1 and 'higher' in text_lower:
            return True

        for sent in doc.sents:
            if sent.root.lemma_ in {'be', 'have'} and any(word.dep_ == 'neg' for word in sent):
                if 'interest' in sent.text or 'accept' in sent.text:
                    return True

        return False

    def _calculate_initial_offer(self) -> float:
        min_salary = float(self.benchmarks.get('min', 4.5)) 
        avg_salary = float(self.benchmarks.get('avg', 7.2))
        max_salary = float(self.benchmarks.get('max', 12.0))
        
        if self.benchmarks.get('source') == 'fallback':
            base = avg_salary * 0.9
        else:
            base = (avg_salary) * 0.9

        skill_factor = min(0.3, len(self.candidate.get_skills_matrix()) * 0.03)
        experience = float(self.candidate.get_experience())
        exp_factor = min(0.2, experience * 0.05)
        weakness_penalty = float(self.candidate.get_weakness_factor()) * 0.05

        offer = base * (1 + skill_factor + exp_factor - weakness_penalty)
        return round(max(offer, (min_salary) * 0.95), 2)

    def _parse_requested_amount(self, text: str) -> Optional[float]:
        doc = nlp_tools.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                cleaned_amount = ''.join(filter(str.isdigit, ent.text))
                if not cleaned_amount:
                    continue
                try:
                    amount = float(cleaned_amount)
                    if 'lakh' in text.lower():
                        return amount * 100000
                    return amount
                except ValueError:
                    continue
        numbers = [float(w.text) for w in doc if w.like_num and w.text.replace('.', '', 1).isdigit()]
        return max(numbers) if numbers else None

    def _calculate_concession(self, candidate_input: str) -> float:
        if not self._detect_negotiation_intent(candidate_input):
            return 0.0
        
        base_concession = 0.03
        adjustments = 0

        doc = nlp_tools.nlp(candidate_input)
        num_words = len(doc)
        sentiment = nlp_tools.sentiment_model(candidate_input)

        if num_words > 15:
            adjustments += 0.01
        if sentiment < -0.2:
            adjustments += 0.02
        elif sentiment > 0.3:
            adjustments -= 0.01

        concession_factor = 1 + (self.concession_count * 0.05)

        return max(0.01, min(0.08, (base_concession + adjustments) * concession_factor))

    def _detect_benefit_requests(self, text: str) -> Dict[str, List[str]]:
        detected = {k: [] for k in Constants.BENEFIT_KEYWORDS.keys()}
        doc = nlp_tools.nlp(text.lower())

        for category, keywords in Constants.BENEFIT_KEYWORDS.items():
            for keyword in keywords:
                for token in doc:
                    if token.lemma_ == keyword:
                        context = " ".join([t.text for t in doc[max(0, token.i - 3):min(len(doc), token.i + 4)]])
                        if context not in detected[category]:
                            detected[category].append(context)

        return {k: v for k, v in detected.items() if v}

    def _generate_response_template(self, analysis: Dict) -> str:
        if self.state == NegotiationState.INITIAL:
            return "Welcome {name}! For the {role} position, we're offering {offer} {currency}. We also provide {initial_benefits_str}."
        elif self.state == NegotiationState.EXPLORING_BENEFITS:
            return "Regarding {benefit_focus}, we can offer {benefit_details}. How does this sound in relation to the salary?"
        elif self.state == NegotiationState.FINAL_OFFER:
            return "Our final offer is {offer} {currency}, including {offered_benefits_str}. Please let us know your decision."
        else:
            return analysis.get('response', {}).get('text', "We're considering your request.")

    def _calculate_new_offer(self, requested: float, concession: float) -> float:
        if self.candidate_request_history:
            max_allowed = min(
                self.current_offer * (1 + Constants.MAX_INCREASE_PCT),
                self.candidate_request_history[-1]
            )
            proposed_offer = self.current_offer * (1 + concession)
            return round(min(proposed_offer, max_allowed), 2)
        
        return round(self.current_offer * (1 + concession), 2)
    
    def _handle_non_negotiation_query(self, candidate_input: str) -> Tuple[float, str]:
        context = {
            'candidate': self.candidate.get_name(),
            'current_offer': self.current_offer,
            'currency': self.benchmarks['currency'],
            'candidate_input': candidate_input,
            'offered_benefits': self.offered_benefits
        }
        
        prompt = (
            "The candidate has accepted the offer and is asking: '{candidate_input}'. "
            "Current offer: {current_offer} {currency}. Benefits: {offered_benefits}. "
            "Generate a helpful HR response without changing numbers."
        ).format(**context)
        
        response = self.llama.analyze_conversation({'prompt': prompt})
        return self.current_offer, response.get('response', {}).get('text', "We'll follow up shortly.")

    def generate_counteroffer(self, candidate_input: str) -> Tuple[float, str]:
        if self.salary_finalized:
            return self._handle_benefits_only(candidate_input)
        
        if self.state == NegotiationState.CLOSED:
            return self.current_offer, "This negotiation is closed."
        
        if self._detect_reject(candidate_input):
            result = self.close_negotiation(accepted=False)
            return result['final_offer'], (
                f"Thank you for your time, {self.candidate.get_name()}. "
                f"We respect your decision and wish you the best in your future endeavors."
            )
            
        if self._detect_acceptance(candidate_input):
            result = self.close_negotiation(accepted=True)
            return result['final_offer'], (
                f"Congratulations {self.candidate.get_name()}! "
                f"We're excited to have you join us at ₹{result['final_offer']}L. "
                "HR will contact you with onboarding details within 24 hours."
            )
            
        is_negotiating = self._detect_negotiation_intent(candidate_input)
    
        if not is_negotiating:
            return self._handle_non_negotiation_query(candidate_input)
        
        if self.state == NegotiationState.CLOSED:
            raise ValueError("Negotiation already closed")

        self.conversation_history.append({'role': 'candidate', 'text': candidate_input, 'timestamp': datetime.now().isoformat()})

        requested_amount = self._parse_requested_amount(candidate_input)
        
        if requested_amount is not None:
            validation_result = self._validate_candidate_request(requested_amount)
            if not validation_result["valid"]:
                return self.current_offer, validation_result["response"]
            
        benefit_requests = self._detect_benefit_requests(candidate_input)

        llm_context = {
            'candidate': self.candidate.get_name(),
            'role': self.role,
            'current_offer': self.current_offer,
            'requested_amount': requested_amount,
            'benefit_requests': benefit_requests,
            'market_min': self.benchmarks['min'],
            'market_max': self.benchmarks['max'],
            'currency': self.benchmarks['currency'],
            'candidate_input': candidate_input,
            'sentiment': nlp_tools.sentiment_model(candidate_input),
            'state': self.state
        }

        analysis = self.llama.analyze_conversation(llm_context)
        hr_response_text = analysis.get('response', {}).get('text', "Let's discuss further.")

        requested_amount = self._parse_requested_amount(candidate_input)
        is_5_percent_request = self._is_within_5_percent_increase(requested_amount)

        concession = self._calculate_concession(candidate_input)
        new_offer = self._calculate_new_offer(requested_amount, concession)

        if is_5_percent_request and new_offer <= requested_amount:
            return self._finalize_salary(new_offer, candidate_input)
        
        if benefit_requests and self.state == NegotiationState.COUNTER_OFFER:
            self.state = NegotiationState.EXPLORING_BENEFITS
            benefit_category = next(iter(benefit_requests))
            self.offered_benefits.setdefault(benefit_category, benefit_requests[benefit_category])

        self.current_offer = new_offer
        self.offer_history.append(new_offer)
        self.concession_count += 1

        if str(new_offer) not in hr_response_text:
            hr_response = f"We're pleased to offer {new_offer} {self.benchmarks['currency']}. {hr_response_text}"
        else:
            hr_response = hr_response_text

        if self.concession_count >= Constants.MAX_CONCESSION_STEPS:
            self.state = NegotiationState.FINAL_OFFER
            hr_response = f"Final Offer: {new_offer} {self.benchmarks['currency']}. {hr_response_text}"

        self.conversation_history.append({'role': 'hr', 'text': hr_response, 'timestamp': datetime.now().isoformat()})
        logging.info(f"HR Response: {hr_response} (Offer: {self.current_offer})")
        return self.current_offer, hr_response
    
    def _is_within_5_percent_increase(self, requested_amount: Optional[float]) -> bool:
        if requested_amount is None:
            return False
        lower_bound = self.current_offer
        upper_bound = round(self.current_offer * 1.05, 2)
        return lower_bound <= requested_amount <= upper_bound

    
    def _finalize_salary(self, accepted_offer: float, candidate_input: str) -> Tuple[float, str]:
        self.salary_finalized = True
        self.current_offer = accepted_offer
        self.offer_history.append(accepted_offer)
        
        benefit_requests = self._detect_benefit_requests(candidate_input)
        response = self._generate_benefits_response(benefit_requests)
        
        return accepted_offer, response
    
    def _generate_benefits_response(self, benefit_requests: Dict) -> str:
        if benefit_requests:
            categories = list(benefit_requests.keys())
            return f"We've agreed on the salary. Let's discuss your {', '.join(categories)} benefits requests."
        return ("We've finalized the salary at {self.current_offer} LPA. "
                "Would you like to discuss additional benefits?")

    def _handle_benefits_only(self, candidate_input: str) -> Tuple[float, str]:
        benefit_requests = self._detect_benefit_requests(candidate_input)
        
        llm_context = {
            'state': NegotiationState.EXPLORING_BENEFITS,
            'benefit_requests': benefit_requests,
            'candidate_input': candidate_input,
            'offered_benefits': self.offered_benefits
        }
        
        analysis = self.llama.analyze_conversation(llm_context)
        hr_response = analysis.get('response', {}).get('text', "Let's discuss benefits.")

        for category, requests in benefit_requests.items():
            self.offered_benefits.setdefault(category, []).extend(requests)

        return self.current_offer, hr_response

    
    def _validate_candidate_request(self, new_request: float) -> Dict:
        validation = {"valid": True, "response": ""}

        if self.candidate_request_history:
            last_request = self.candidate_request_history[-1]
            
            if new_request > last_request:
                validation["valid"] = False
                validation["response"] = (
                    f"Our records show your previous request was {last_request} LPA. "
                    "We cannot consider higher amounts at this stage of negotiation."
                )
                return validation

        self.candidate_request_history.append(new_request)
        return validation
    
    def _format_benefits(self, benefits: Dict[str, List[str]]) -> str:
        if not benefits:
            return "our standard benefits package"
        formatted = []
        for category, requests in benefits.items():
            examples = ", ".join(requests[:2])
            formatted.append(f"potential for {category.title()} benefits like {examples}")
        return "including " + " and ".join(formatted) if formatted else "our comprehensive benefits package"

    def close_negotiation(self, accepted: bool) -> Dict:
        self.state = NegotiationState.CLOSED
        self.context_window = []
        
        if accepted:
            self.concession_count = 0

        result = {
            'final_offer': self.current_offer,
            'status': 'accepted' if accepted else 'declined',
            'concession_steps': self.concession_count,
            'offer_history': self.offer_history,
            'offered_benefits': self.offered_benefits,
            'conversation_history': self.conversation_history,
            'timestamp': datetime.now().isoformat()
        }
        logging.info(f"Negotiation closed: {result}")
        return result

class SmartHRInterface:
    def __init__(self, report_path: str):
        self.candidate = CandidateProfile(report_path)
        self.role=""
        self.engine = None

    def temp_func(self,role):
        self.role=role
        self.engine = NegotiationEngine(self.candidate, self.role)
        self._setup()
        
    def _setup(self):
        Constants.CACHE_DIR.exists() or Constants.CACHE_DIR.mkdir()
        return True

    def _select_role(self) -> list:
        roles = self.candidate.get_roles()
        if not roles:
            raise ValueError("No roles found in candidate report. Cannot proceed with negotiation.")
        
        return roles
       

    def _display_offer_details(self) -> Dict:
        if not self.engine:
            raise ValueError("Negotiation engine not initialized")
        return {
            "candidate_name": self.candidate.get_name(),
            "position": self.role,
            "initial_offer": f"{self.engine.current_offer} LPA",
            "market_range": f"{self.engine.benchmarks['min']}-{self.engine.benchmarks['max']} LPA",
            "development_benefits": self.candidate.hr_strategies.get('benefit_adjustments', []),
            "message": "What are your initial thoughts?"
        }

    def conduct_negotiation(self) -> Dict:
        negotiation_result = {
            "initial_offer": self._display_offer_details(),
            "conversation": [],
            "final_result": None
        }

        while self.engine.state != NegotiationState.CLOSED:
            try:
                candidate_response = {"input": "Sample input", "timestamp": datetime.now().isoformat()}
                
                offer, hr_response = self.engine.generate_counteroffer(candidate_response["input"])
                negotiation_result["conversation"].append({
                    "candidate": candidate_response,
                    "hr": {
                        "response": hr_response,
                        "current_offer": f"{offer} LPA",
                        "timestamp": datetime.now().isoformat()
                    }
                })

                if self.engine.state == NegotiationState.FINAL_OFFER:
                    decision = "accept"  # In actual implementation, this would come from input
                    result = self.engine.close_negotiation(decision == 'accept')
                    negotiation_result["final_result"] = self._display_final_result(result)
                    break

            except KeyboardInterrupt:
                negotiation_result["final_result"] = {"status": "interrupted", "message": "Negotiation interrupted by user"}
                break
            except Exception as e:
                logging.error(f"Negotiation error: {str(e)}", exc_info=True)
                negotiation_result["final_result"] = {"status": "error", "message": "We're experiencing technical difficulties. Please try again later."}
                break

        return negotiation_result

    def _display_final_result(self, result: Dict) -> Dict:
        final_result = {
            "final_offer": f"{result['final_offer']} {self.engine.benchmarks['currency']}",
            "status": result['status'].upper(),
            "agreed_benefits": [],
            "conversation_history": [],
            "next_steps": ""
        }

        if result['offered_benefits']:
            for category, requests in result['offered_benefits'].items():
                final_result["agreed_benefits"].append({
                    "category": category.title(),
                    "details": requests[:2]
                })

        for turn in result['conversation_history']:
            final_result["conversation_history"].append({
                "participant": turn['role'].upper(),
                "timestamp": turn['timestamp'].split('.')[0],
                "message": turn['text']
            })

        if result['status'] == 'accepted':
            final_result["next_steps"] = "HR will contact you within 24 hours for onboarding."
        else:
            final_result["next_steps"] = "We appreciate your time and consideration."

        return final_result

def main():
    try:
        # Initialize necessary components
        expert = LlamaNegotiationExpert()
        print("=== Advanced HR Negotiation System ===")
        report_path = r"C:\Users\athis\OneDrive\Documents\PSG\Project\Payroll System\app\FullAnalysis\Athish_S_R.json"
        if not Path(report_path).exists():
            raise FileNotFoundError(f"Report not found: {report_path}")
        
        session = SmartHRInterface(report_path)
        
        roles = session._select_role()
        print("\nAvailable positions:")
        for idx, role in enumerate(roles, 1):
            print(f"{idx}. {role}")
        
       
        choice = input("\nPlease select a position number or 'q' to quit: ").strip()
        if choice.lower() == 'q':
            raise SystemExit("Negotiation cancelled by user.")
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(roles):
                session.temp_func(roles[idx])
                    

        
        # Display initial offer details
        initial_details = session._display_offer_details()
        print(f"\nHR: Welcome {initial_details['candidate_name']}!")
        print(f"Position: {initial_details['position']}")
        print(f"Initial Offer: {initial_details['initial_offer']}")
        print(f"Market Range: {initial_details['market_range']}")
        print("Development Benefits:")
        for benefit in initial_details['development_benefits']:
            print(f"- {benefit.get('type', 'Benefit')}: {benefit.get('recommendation', '')}")
        print(initial_details['message'])
        
        # Negotiation loop
        while True:
            candidate_input = input("\nCandidate: ").strip()
            
            if candidate_input.lower() == 'q':
                print("Negotiation cancelled by user.")
                break
            
            # Process candidate input and get HR response
            current_offer, hr_response = session.engine.generate_counteroffer(candidate_input)
            print(f"\nHR: {hr_response}")
            
            # Check if negotiation is closed
            if session.engine.state == NegotiationState.CLOSED:
                # Determine negotiation outcome
                status = 'accepted' if 'congratulations' in hr_response.lower() else 'declined'
                final_offer = session.engine.current_offer
                offered_benefits = session.engine.offered_benefits
                
                print("\n=== Negotiation Result ===")
                print(f"Final Offer: {final_offer} LPA")
                print(f"Status: {status.upper()}")
                if offered_benefits:
                    print("Agreed Benefits:")
                    for category, requests in offered_benefits.items():
                        print(f"- {category.title()}: {', '.join(requests[:2])}")
                if status == 'accepted':
                    print("\nNext Steps: HR will contact you within 24 hours for onboarding.")
                else:
                    print("\nThank you for your time. We wish you the best in your future endeavors.")
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("\nSession ended.")

if __name__ == "__main__":
    main()