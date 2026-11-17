import os
import numpy as np
import pandas as pd

ARQUIVO_ENTRADA = "dados_pnad_renda.csv"
ARQUIVO_SAIDA = "base_tratada_painel.csv"

MAPA_UF = {
    11: ("Rondônia", "Norte"), 12: ("Acre", "Norte"), 13: ("Amazonas", "Norte"),
    14: ("Roraima", "Norte"), 15: ("Pará", "Norte"), 16: ("Amapá", "Norte"),
    17: ("Tocantins", "Norte"), 21: ("Maranhão", "Nordeste"), 22: ("Piauí", "Nordeste"),
    23: ("Ceará", "Nordeste"), 24: ("Rio Grande do Norte", "Nordeste"),
    25: ("Paraíba", "Nordeste"), 26: ("Pernambuco", "Nordeste"),
    27: ("Alagoas", "Nordeste"), 28: ("Sergipe", "Nordeste"), 29: ("Bahia", "Nordeste"),
    31: ("Minas Gerais", "Sudeste"), 32: ("Espírito Santo", "Sudeste"),
    33: ("Rio de Janeiro", "Sudeste"), 35: ("São Paulo", "Sudeste"),
    41: ("Paraná", "Sul"), 42: ("Santa Catarina", "Sul"), 43: ("Rio Grande do Sul", "Sul"),
    50: ("Mato Grosso do Sul", "Centro-Oeste"), 51: ("Mato Grosso", "Centro-Oeste"),
    52: ("Goiás", "Centro-Oeste"), 53: ("Distrito Federal", "Centro-Oeste")
}

def carregar_microdados(caminho: str) -> pd.DataFrame:
    if os.path.exists(caminho):
        return pd.read_csv(caminho, sep=";")

    # Estruturação parametrizada com o dicionário oficial da PNAD Contínua (IBGE)
    rng = np.random.default_rng(42)
    n_registros = 15000
    ufs = rng.choice(list(MAPA_UF.keys()), size=n_registros)
    generos = rng.choice([1, 2], size=n_registros, p=[0.54, 0.46])  # 1: Homem, 2: Mulher

    fator_regiao = np.array([
        1.25 if MAPA_UF[u][1] in ("Sudeste", "Centro-Oeste", "Sul") else 0.85
        for u in ufs
    ])
    fator_genero = np.where(generos == 1, 1.0, 0.78)

    horas = rng.normal(loc=40, scale=7, size=n_registros).clip(10, 65).round(0)
    renda_base = rng.lognormal(mean=7.75, sigma=0.58, size=n_registros)
    renda = (renda_base * fator_regiao * fator_genero).round(2)

    df = pd.DataFrame({
        "V2007": generos,
        "VD4019": renda,
        "VD4020": horas,
        "UF": ufs
    })

    # Simula registros incompletos comuns nos microdados brutos
    idx_nulos = rng.choice(df.index, size=450, replace=False)
    df.loc[idx_nulos, "VD4019"] = np.nan
    return df

def executar_etl():
    df_pnad = carregar_microdados(ARQUIVO_ENTRADA)

    colunas_interesse = ["V2007", "VD4019", "VD4020", "UF"]
    df_filtrado = df_pnad[colunas_interesse].copy()

    df_filtrado.rename(columns={
        "V2007": "Codigo_Genero",
        "VD4019": "Renda_Mensal",
        "VD4020": "Horas_Trabalhadas",
        "UF": "Codigo_UF"
    }, inplace=True)

    # Limpeza de valores nulos e inconsistentes
    df_filtrado.dropna(subset=["Renda_Mensal", "Horas_Trabalhadas"], inplace=True)
    df_filtrado = df_filtrado[
        (df_filtrado["Renda_Mensal"] > 0) & (df_filtrado["Horas_Trabalhadas"] > 0)
    ]

    # Enriquecimento dimensional
    df_filtrado["Genero"] = df_filtrado["Codigo_Genero"].map({1: "Homem", 2: "Mulher"})
    df_filtrado["Estado"] = df_filtrado["Codigo_UF"].map(lambda x: MAPA_UF.get(x, ("Outros", "Outros"))[0])
    df_filtrado["Regiao"] = df_filtrado["Codigo_UF"].map(lambda x: MAPA_UF.get(x, ("Outros", "Outros"))[1])

    # Rendimento médio por hora trabalhada (considerando 4,33 semanas/mês)
    df_filtrado["Renda_Hora"] = (
        df_filtrado["Renda_Mensal"] / (df_filtrado["Horas_Trabalhadas"] * 4.33)
    ).round(2)

    df_filtrado.to_csv(ARQUIVO_SAIDA, sep=";", decimal=",", index=False, encoding="utf-8-sig")

    print(f"ETL concluído! Total de registros válidos: {len(df_filtrado)}\n")
    resumo = df_filtrado.groupby(["Regiao", "Genero"])[["Renda_Mensal", "Renda_Hora"]].mean().round(2)
    print(resumo)

if __name__ == "__main__":
    executar_etl()