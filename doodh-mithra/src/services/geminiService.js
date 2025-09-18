import { GoogleGenerativeAI } from '@google/generative-ai';

// Initialize Gemini AI with the provided API key
const GEMINI_API_KEY = '';
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

// Helper function to provide language-specific examples
const getLanguageExample = (language) => {
  const examples = {
    'ta': `
      உதாரணம்:
      1. பால் உற்பத்தி விவரக்குறிப்பு:
         - தினசரி பால் உற்பத்தி: 15-20 லிட்டர்
         - பால் கொழுப்பு சதவீதம்: 4.5%`,
    'hi': `
      उदाहरण:
      1. दूध उत्पादन प्रोफ़ाइल:
         - दैनिक दूध उत्पादन: 15-20 लीटर
         - दूध में वसा का प्रतिशत: 4.5%`,
    'en': `
      Example:
      1. MILK PRODUCTION PROFILE:
         - Daily milk yield: 15-20 liters
         - Milk fat percentage: 4.5%`,
    'bn': `
      উদাহরণ:
      1. দুধ উৎপাদন প্রোফাইল:
         - দৈনিক দুধ উৎপাদন: ১৫-২০ লিটার
         - দুধে চর্বির শতাংশ: ৪.৫%`,
    'te': `
      ఉదాహరణ:
      1. పాల ఉత్పాదన ప్రొఫైల్:
         - రోజువారీ పాల ఉత్పాదన: 15-20 లీటర్లు
         - పాలలో కొవ్వు శాతం: 4.5%`,
    'kn': `
      ಉದಾಹರಣೆ:
      1. ಹಾಲು ಉತ್ಪಾದನೆ ಪ್ರೊಫೈಲ್:
         - ದೈನಂದಿನ ಹಾಲು ಉತ್ಪಾದನೆ: 15-20 ಲೀಟರ್
         - ಹಾಲಿನಲ್ಲಿ ಕೊಬ್ಬಿನ ಶೇಕಡಾವಾರು: 4.5%`
  };
  return examples[language] || examples['en'];
};

// Helper function to get breed name in local language
const getBreedNameInLanguage = (breedName, language) => {
  const breedTranslations = {
    'Jersey': {
      'ta': 'ஜெர்சி',
      'hi': 'जर्सी',
      'bn': 'জার্সি',
      'te': 'జెర్సీ',
      'kn': 'ಜರ್ಸಿ'
    },
    'Holstein': {
      'ta': 'ஹோல்ஸ்டீன்',
      'hi': 'होल्स्टीन',
      'bn': 'হোলস্টেইন',
      'te': 'హోల్‌స్టీన్',
      'kn': 'ಹೋಲ್‌ಸ್ಟೈನ್'
    },
    'Gir': {
      'ta': 'கிர்',
      'hi': 'गिर',
      'bn': 'গির',
      'te': 'గిర్',
      'kn': 'ಗಿರ್'
    },
    'Sahiwal': {
      'ta': 'சாஹிவால்',
      'hi': 'साहीवाल',
      'bn': 'সাহিওয়াল',
      'te': 'సాహివాల్',
      'kn': 'ಸಾಹಿವಾಲ್'
    }
  };
  
  return breedTranslations[breedName]?.[language] || breedName;
};

export const getBreedDetails = async (breedName, breedType, language = 'ta') => {
  try {
    console.log('Fetching breed details for:', breedName, 'in language:', language);
    
    // Get the generative model - using Gemini 2.5 Flash
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-exp" });

    // Language mapping for prompts
    const languageNames = {
      'ta': 'Tamil',
      'en': 'English', 
      'hi': 'Hindi',
      'bn': 'Bangla',
      'te': 'Telugu',
      'kn': 'Kannada'
    };

    const selectedLanguageName = "English";

    // Create a comprehensive prompt for cattle breed information
    const prompt = `
      You are an expert dairy veterinarian and livestock consultant. I need comprehensive information about the "${breedName}" cattle breed in ${selectedLanguageName} language.

      CRITICAL LANGUAGE INSTRUCTIONS - READ CAREFULLY:
      1. Write the ENTIRE response in English language only
      2. ALL section headings must be in English 
      3. ALL content, explanations, and recommendations must be in English 
      4. Use proper English  script/alphabet for the language
      5. Do NOT mix languages - everything should be in English 
      6. Numbers and measurements can remain in standard format but with English  text labels
      7. At the beginning, state the breed name as: "[Breed name in English ] (English: ${breedName})"

      Focus specifically on MILK PRODUCTION and DISEASE RESISTANCE to help farmers make smart breeding decisions and improve productivity and financial outcomes.

      Please structure your response with these sections in English :

      1. MILK PRODUCTION PROFILE (in English ):
         - Daily milk yield (liters per day)
         - Lactation period duration
         - Peak milk production timing
         - Milk fat percentage and quality
         - Annual milk production capacity
         - Factors affecting milk yield

      2. DISEASE RESISTANCE (in English ):
         - Natural immunity levels of this breed
         - Specific diseases this breed is resistant to
         - Genetic resistance factors
         - Climate adaptability and disease tolerance
         - Comparison with other breeds for disease resistance

      3. HEALTH MANAGEMENT (in $English ):
         - Common diseases this breed is susceptible to
         - Vaccination schedule and prevention protocols
         - Health indicators to monitor
         - Cost-effective health management strategies
         - Emergency health protocols

      4. BREEDING RECOMMENDATIONS FOR PRODUCTIVITY (in English ):
         - Best breeding age and timing
         - Genetic selection criteria for high milk yield
         - Breeding cycle optimization
         - Crossbreeding options for improved production
         - Signs of good breeding stock

      5. FINANCIAL ANALYSIS AND PROFITABILITY ROADMAP (in English ):
         YEAR 1 SETUP:
         - Initial investment breakdown: cattle purchase, housing, equipment
         - Setup costs: feeding systems, milking equipment, veterinary setup
         - First year revenue expectations and cash flow
         - Government loans and subsidies available
         
         YEAR 2-3 GROWTH PHASE:
         - Monthly operational costs (feed, veterinary, labor)
         - Break-even analysis and profit projections
         - Expansion planning: when to add more cattle
         - Risk management and insurance planning
         
         YEAR 4+ EXPANSION:
         - Long-term profitability analysis
         - Value-added opportunities (cheese, butter, etc.)
         - Market diversification strategies
         - Retirement and exit planning
         
         FINANCIAL ROADMAP MILESTONES:
         - Month 6: First calving and milk production
         - Year 1: Break-even achievement
         - Year 2: 15-20% profit margin target
         - Year 3+: Sustainable growth and expansion

      6. SMART FARMING DECISIONS (in ${selectedLanguageName}):
         - When to buy vs breed this cattle
         - Optimal herd size for profitability
         - Seasonal considerations for maximum profit
         - Feed cost optimization strategies
         - When to cull for better productivity

      7. PRODUCTIVITY ENHANCEMENT TIPS (in ${selectedLanguageName}):
         - Nutrition requirements for maximum milk yield
         - Housing conditions for optimal production
         - Milking techniques and frequency
         - Stress reduction for better performance
         - Technology integration opportunities

      8. ECONOMIC DECISION FACTORS (in ${selectedLanguageName}):
         - Market timing for selling milk/cattle
         - Value-added products opportunities
         - Insurance and risk management
         - Government schemes and subsidies available
         - Long-term vs short-term profitability

      IMPORTANT FORMATTING REQUIREMENTS:
      - Use ${selectedLanguageName} language for ALL text
      - Start with breed name in both languages: "[Breed Name in ${selectedLanguageName}] (English: ${breedName})"
      - Provide specific numbers, timeframes, and actionable recommendations
      - Focus on helping farmers increase milk production, reduce disease-related losses, and make financially sound breeding decisions
      - Use simple language without technical jargon that rural farmers can easily understand
      - Write section headings in ${selectedLanguageName}
      
      EXAMPLE FORMAT FOR ${selectedLanguageName}:
      ${getLanguageExample(language)}
      
      FINAL REMINDER: The complete response must be written in ${selectedLanguageName} language. Do not use English except for the breed name reference.
    `;

    // Generate content
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();

    console.log('Gemini API response received');
    
    // Parse the response into structured data
    const sections = parseBreedInformation(text);
    
    return {
      success: true,
      breedName,
      breedType,
      information: text,
      sections: sections,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    console.error('Error fetching breed details from Gemini:', error);
    
    // Return fallback information
    return {
      success: false,
      error: error.message,
      breedName,
      breedType,
      information: `Unable to fetch detailed breed information. Please try again later.`,
      sections: {},
      timestamp: new Date().toISOString()
    };
  }
};

// Helper function to parse the Gemini response into structured sections
const parseBreedInformation = (text) => {
  const sections = {};
  
  try {
    // Clean up the text first - remove ** symbols and format properly
    let cleanedText = text
      .replace(/\*\*(.*?)\*\*/g, '$1') // Remove ** around text
      .replace(/\*\*/g, '') // Remove any remaining **
      .replace(/\*/g, '•'); // Convert * to bullet points
    
    // Split by numbered sections or headers
    const lines = cleanedText.split('\n');
    let currentSection = '';
    let currentContent = [];
    
    lines.forEach(line => {
      const trimmedLine = line.trim();
      
      // Check for section headers (numbered sections, all caps, or clear headers)
      if (trimmedLine.match(/^\d+\.\s*[A-Z][A-Z\s:]+/) || 
          trimmedLine.match(/^[A-Z][A-Z\s:]{10,}/) ||
          trimmedLine.match(/^\d+\.\s*\w+/)) {
        
        // Save previous section
        if (currentSection && currentContent.length > 0) {
          sections[currentSection] = currentContent.join('\n').trim();
        }
        
        // Start new section
        currentSection = trimmedLine.replace(/^\d+\.\s*/, '').replace(/:/g, '').trim();
        currentContent = [];
      } else if (trimmedLine && currentSection) {
        currentContent.push(trimmedLine);
      }
    });
    
    // Save last section
    if (currentSection && currentContent.length > 0) {
      sections[currentSection] = currentContent.join('\n').trim();
    }
    
  } catch (error) {
    console.error('Error parsing breed information:', error);
  }
  
  return sections;
};