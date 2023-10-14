import requests
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv


def get_location():
    url = 'http://ip-api.com/json/'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        country = data.get('country', 'N/A')
        assert 'United States' == country, f'Country is not United States: {country}'
    else:
        print(f'Failed to retrieve location. Status code: {response.status_code}')


def generate_pet_name(animal_type='dog', temperature=0.9):
    llm = OpenAI(temperature=temperature)
    prompt_template_name = PromptTemplate(
        input_variables=['animal_type'],
        template='I have a {animal_type} pet and i want a cool name for it.Suggest me five cool name for my pet.',
    )
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name)
    response = name_chain({'animal_type': animal_type})
    print(response)


if __name__ == "__main__":
    load_dotenv()
    generate_pet_name()
