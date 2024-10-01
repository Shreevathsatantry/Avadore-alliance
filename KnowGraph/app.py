%%writefile app.py
import streamlit as st
import pandas as pd
from neo4j import GraphDatabase
from streamlit_agraph import agraph, Config, Node, Edge
from config import ConfigBuilder
import requests
import json
from AutoClean import AutoClean
from langchain.schema import Document
from langchain.document_loaders import CSVLoader 
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.llms import Ollama
from langchain_community.graphs import Neo4jGraph

driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "password"))

def visualize_graph():
    nodes = []
    edges = []
    seen_nodes = set()
    with driver.session() as session:
        result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m")
        for record in result:
            node1_id = str(record["n"].id)
            node2_id = str(record["m"].id)
            rel_type = record["r"].type

            if node1_id not in seen_nodes:
                nodes.append(Node(id=node1_id, label=record["n"].get("name", node1_id)))
                seen_nodes.add(node1_id)
            if node2_id not in seen_nodes:
                nodes.append(Node(id=node2_id, label=record["m"].get("name", node2_id)))
                seen_nodes.add(node2_id)

            edges.append(Edge(source=node1_id, target=node2_id, label=rel_type))

    if not nodes:
        return None

    config_builder = ConfigBuilder(nodes=nodes, edges=edges)
    config = config_builder.build()

    return agraph(nodes=nodes, edges=edges, config=config)

ollama_url = "https://abaa-34-125-207-21.ngrok-free.app"
llm = Ollama(model="llama3", base_url=ollama_url)

st.title("RAG Graph with Neo4j, Llama3 (Ollama), and CSV Upload")

uploaded_file = st.file_uploader("Choose a file", type=['csv'])

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    st.subheader("Uploaded file")
    with st.expander("View Uploaded Data"):
        st.dataframe(dataset)

    st.subheader("Deleting existing Neo4j nodes and relationships")
    with st.spinner("Deleting existing graph data..."):
        query = "MATCH (n) DETACH DELETE n"
        try:
            with driver.session() as session:
                session.run(query)
            st.success("Successfully deleted all existing nodes and relationships in Neo4j.")
        except Exception as e:
            st.error(f"An error occurred while deleting nodes: {e}")

    st.subheader("Cleaned and converted data")
    with st.spinner("Cleaning and converting to schema..."):
        pipeline = AutoClean(dataset)
        txt_file = "cleaned_customers.csv"
        pipeline.output.to_csv(txt_file, sep=",", index=True, header=True)
        
        loader = CSVLoader(file_path="cleaned_customers.csv")
        docs = loader.load()
        documents = [Document(page_content=doc.page_content) for doc in docs]
        
        st.write("First few converted documents:")
        for doc in documents[:10]:
            st.write(doc.page_content)

    if 'graph_documents' not in st.session_state:
        st.session_state.graph_documents = None

    st.subheader("Graph Conversion")
    with st.spinner("Converting to graph document..."):
        if st.session_state.graph_documents is None:
            llm_transformer = LLMGraphTransformer(llm=llm)
            st.session_state.graph_documents = llm_transformer.convert_to_graph_documents(documents)
        graph_documents = st.session_state.graph_documents

    if graph_documents:
        with st.expander("Nodes and Relations"):
            st.subheader("Nodes")
            for node in graph_documents[0].nodes:
                st.write(node)

            st.subheader("Relationships")
            for relation in graph_documents[0].relationships:
                st.write(relation)

    st.subheader("Adding new graph documents to Neo4j")
    with st.spinner("Adding data to Neo4j..."):
        try:
            graph = Neo4jGraph(url="neo4j://localhost:7687", username="neo4j", password="password")
            graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )
            st.success("Successfully added new graph documents to Neo4j.")
        except Exception as e:
            st.error(f"An error occurred while adding graph documents: {e}")

    st.subheader("Graph Visualization")
    graph_visualization = visualize_graph()
    if graph_visualization:
        st.write(graph_visualization)
    else:
        st.warning("No data available for visualization.")
