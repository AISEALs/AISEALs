import datetime


def date_range(start_date, end_date):
    for i in range((end_date - start_date).days):
        day = start_date + datetime.timedelta(days=i)
        yield day.strftime("%Y%m%d")


if __name__ == '__main__':
    start_date = datetime.date(2021, 3, 1)
    end_date = datetime.date(2021, 3, 24)
    date_list = [int(d) for d in date_range(start_date, end_date)]

    day_hour = datetime.datetime.strptime(str(2022050114), '%Y%m%d%H')
    print(day_hour.hour)
    print(day_hour.date())
    print(day_hour.strftime("%Y%m%d%H"))

