from agents.hospital_rag_agent import hospital_agent_excutor

response = hospital_agent_excutor.invoke({"input": ("Which physician has treated the"
                                                    "most patients covered by Cigna?")})
print(response.get('output'))