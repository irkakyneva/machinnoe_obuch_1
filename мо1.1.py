
r"""
Запуск:
    python "C:\Users\irkak\Downloads\cars_analysis.py"
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

REQUIRED_COLUMNS = {"mileage", "make", "model", "fuel", "gear", "offerType", "price", "hp", "year"}


def print_header() -> None:
    print("=" * 90)
    print("Семинар №1".center(90))
    print("Дисциплина: Машинное обучение".center(90))
    print("Выполнила: Кынева Ирина".center(90))
    print("Группа: ИДБ-24-15".center(90))
    print("=" * 90)


def title(text: str) -> None:
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)


def find_csv(cli_path):
    if cli_path:
        p = Path(cli_path)
        if not p.exists():
            sys.exit(f"файл не найден: {p}")
        return p
    here = Path(__file__).resolve().parent
    for folder in (Path.cwd(), here):
        for name in ("cars_1.csv", "cars.csv"):
            if (folder / name).exists():
                return folder / name
    sys.exit("не найден cars_1.csv / cars.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="анализ датасета cars.csv")
    parser.add_argument("csv", nargs="?", help="путь к CSV-файлу")
    parser.add_argument("--show", action="store_true", help="показать графики на экране")
    parser.add_argument("--out", default="cars_hist.png", help="файл для сохранения гистограмм")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print_header()


    # ------------------------------------- 1 загрузка данных

    title("1 ЗАГРУЗКА ДАННЫХ")
    path = find_csv(args.csv)
    df = pd.read_csv(path)
    print(f"Файл: {path}")
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        sys.exit(f"в файле нет ожидаемых столбцов: {sorted(missing_cols)}")
    print(df.head())


    # ---------------------------------------2 первичный осмотр

    title("2 ПЕРВИЧНЫЙ ОСМОТР ДАННЫХ")
    print(f"размер датафрейма: {df.shape[0]} строк, {df.shape[1]} столбцов\n")
    df.info()

    print("\n--- числовые признаки ---")
    print(df.describe())
    print("\n--- категориальные признаки ---")
    print(df.describe(include=["object", "string"]))

    for col in ("fuel", "gear", "offerType"):
        print(f"\nзначения столбца {col}:")
        print(df[col].value_counts(dropna=False))

    num_cols = ["mileage", "price", "hp", "year"]
    axes = df[num_cols].hist(bins=40, figsize=(12, 8))
    plt.suptitle("распределения числовых признаков")
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"\nгистограммы сохранены в файл: {Path(args.out).resolve()}")
    if args.show:
        plt.show()
    plt.close("all")

    stats = df[num_cols].describe()
    print("\nВЫВОДЫ ПО ПЕРВИЧНОМУ ОСМОТРУ:")
    print(f"  * в датасете {df.shape[0]} строк и {df.shape[1]} столбцов: 4 числовых и 5 категориальных.")
    na_now = df.isna().sum()
    na_now = na_now[na_now > 0]
    print("  * пропуски есть в столбцах: " + ", ".join(f"{c} ({n})" for c, n in na_now.items()) + ".")
    print(f"  * годы выпуска: {int(stats.loc['min', 'year'])}-{int(stats.loc['max', 'year'])}; "
          f"медианный пробег {stats.loc['50%', 'mileage']:,.0f} км; медианная цена {stats.loc['50%', 'price']:,.0f}.")
    print(f"  * распределения mileage, price, hp скошены вправо; есть выбросы: "
          f"пробег до {stats.loc['max', 'mileage']:,.0f} км, цена до {stats.loc['max', 'price']:,.0f}, "
          f"мощность до {stats.loc['max', 'hp']:,.0f} л.с.")
    print("  * price и hp имеют тип float только из-за пропусков (NaN).")
    print("  * В fuel значения '-/- (Fuel)' и 'Others' фактически означают 'неизвестно / другое'.")


    # ---------------------------------3 переименование offerType

    title("3 ПЕРЕИМЕНОВАНИЕ СТОЛБЦА offerType")
    df = df.rename(columns={"offerType": "offer_type"})
    print("столбцы:", df.columns.tolist())


    # ----------------------------------4 дубликаты

    title("4 РАБОТА С ДУБЛИКАТАМИ")
    n_dup = int(df.duplicated().sum())
    print(f"количество дублированных строк: {n_dup} ({n_dup / len(df) * 100:.1f}% данных)")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"дубликатов после удаления: {int(df.duplicated().sum())}")
    print(f"размер датафрейма после очистки: {df.shape}")
    print("вывод : полные дубли -- вероятно одно и то же объявление, выгруженное несколько раз; "
          "они искажают статистику поэтому удалены > индекс сброшен")


    # --------------------------------5 пропуски

    title("5 ПРОПУСКИ В ДАННЫХ")
    miss = df.isna().sum()
    print(pd.DataFrame({"пропусков": miss, "доля, %": (miss / len(df) * 100).round(2)}))
    print("""
описание пропусков:
  * price - больше всего пропусков. Продавец мог не указать цену ("договорная", "по запросу")
    либо она не распарсилась. Исправление: удалить строки (цена - целевой признак) или заполнить
    медианой по группе (make + model + year).
  * fuel, gear, model - поле не заполнено продавцом или не распознано при сборе данных.
    Исправление: мода по марке/модели, значение "Unknown" или оставить как есть.
  * hp - единичные пропуски, мощность не указана. Исправление: медиана по модели и году выпуска.""")


    # -----------------------------6-7 Удаление пропусков в hp и price + проверка

    title("6-7 УДАЛЕНИЕ ПРОПУСКОВ В hp И price И ПРОВЕРКА")
    before = len(df)
    df = df.dropna(subset=["hp", "price"]).reset_index(drop=True)
    print(f"удалено строк: {before - len(df)}; осталось: {len(df)}")
    print("\nпропусков в hp и price после удаления:")
    print(df[["hp", "price"]].isna().sum())
    ok = not df[["hp", "price"]].isna().any().any()
    print(f"Все пропуски в hp и price удалены: {ok}")
    print("\nОставшиеся пропуски в других столбцах:")
    print(df.isna().sum())


    # 8 ------------------------ количество уникальных марок

    title("8 КОЛИЧЕСТВО УНИКАЛЬНЫХ МАРОК")
    print(f"уникальных марок : {len(df['make'].unique())}")


    # 9----------------------------- топ-20 марок по числу объявлений

    title("9 ТОП-20 МАРОК ПО КОЛИЧЕСТВУ ОБЪЯВЛЕНИЙ")
    top20 = df["make"].value_counts().head(20)
    print(top20)
    print(f"\nвывод: лидер - {top20.index[0]} ({top20.iloc[0]} объявлений), далее "
          f"{', '.join(top20.index[1:5])}. В топе преобладают массовые европейские бренды")

    #----------------------------------10 группировка по make

    title("10 СРЕДНЯЯ ЦЕНА И МОЩНОСТЬ ПО ПРОИЗВОДИТЕЛЯМ (ТОП-20 САМЫХ ДОРОГИХ)")
    top_make = (df.groupby("make")[["price", "hp"]]
                  .mean()
                  .sort_values("price", ascending=False)
                  .head(20))
    top_make["объявлений"] = df.groupby("make").size()[top_make.index]
    print(top_make)

    small = top_make[top_make["объявлений"] < 5].index.tolist()
    print("\nВЫВОД:")
    print("  *рейтинг в целом логичен: наверху люксовые бренды и спорткары с мощностью 500-700 л.с.")
    print(f"  * смущает: у части брендов очень мало объявлений (<5): {', '.join(small)}. "
          "среднее по ним ненадёжно - одна машина определяет результат")
    print("  * Названия 'Aston' и 'Land' обрезаны по первому пробелу (Aston Martin, Land Rover) - "
          "значения в make нужно нормализовать.")
    print("  * среднее чувствительно к выбросам; для сравнения брендов лучше медиана + фильтр по числу объявлений.")


    #--------------------------------- 11 make + model, топ-10 дорогих и дешёвых

    title("11. ТОП-10 САМЫХ ДОРОГИХ И ДЕШЁВЫХ АВТОМОБИЛЕЙ (make + model)")
    df["make_model"] = df["make"] + " " + df["model"]
    ms = (df.groupby("make_model")["price"]
            .agg(avg_price="mean", n_ads="count"))

    print("топ-10 самых дорогих:")
    print(ms.sort_values("avg_price", ascending=False).head(10))
    print("\nтоп-10 самых дешёвых:")
    print(ms.sort_values("avg_price").head(10))

    print("\nДополнительно - то же, но только для моделей с >= 10 объявлениями (более надёжная оценка):")
    reliable = ms[ms["n_ads"] >= 10]
    print("\nсамые дорогие:")
    print(reliable.sort_values("avg_price", ascending=False).head(10))
    print("\nсамые дешёвые:")
    print(reliable.sort_values("avg_price").head(10))
    print("\nвывод: в обычных топах многие позиции держатся на 1-3 объявлениях, поэтому они информативны "
          "лишь условно; фильтр по числу объявлений даёт более устойчивый рейтинг.")


    # 12 -------------------------------Общий вывод

    title("12 ОБЩИЙ ВЫВОД")
    print(f"""\
1. Исходные данные: объявления с 9 признаками. Выявлены пропуски (price, fuel, gear, model, hp),
   скошенные распределения mileage, price, hp и явные выбросы.
2. Предобработка: offerType переименован в offer_type; удалено {n_dup} полных дубликатов;
   удалены строки с пропусками в hp и price. Итоговый датасет: {len(df)} строк.
3. В данных {df['make'].nunique()} уникальных марок; по числу объявлений лидирует {top20.index[0]}.
4. Самые дорогие марки/модели - люксовые автомобили и спорткары, самые дешёвые - бюджетные городские
   модели. Средние для редких марок и моделей опираются на 1-3 объявления и искажены выбросами.
5. Дальше стоит: исправить обрезанные названия марок (Aston -> Aston Martin, Land -> Land Rover),
   обработать выбросы (пробег ~1.1 млн км, цены ~1 млн), заполнить пропуски в категориальных признаках
   и сравнивать бренды по медиане с фильтром по числу объявлений.""")


if __name__ == "__main__":
    main()
